"""Train or evaluate BC-SSM and its four conditioning configurations."""

import argparse
import json
import random
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from neuralhydrology.evaluation.metrics import calculate_metrics
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import CamelsWindows, OUTPUT_MEAN, OUTPUT_STD, attributes, static_scaler
from model import SelectiveSSMConditioned, SelectiveSSMInputOnly

MODES = {"bc_ssm": "both", "state_only": "state_only", "readout_only": "readout_only"}
MODELS = ["bc_ssm", "no_static", "concat_static", "state_only", "readout_only"]
METRICS = ["NSE", "KGE", "Pearson-r", "FHV", "FLV", "Alpha-NSE", "Beta-NSE", "Beta-KGE", "RMSE"]


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_config(path: Path, model=None, setup=None, split=None) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for name, value in (("model", model), ("setup", setup), ("split", split)):
        if value is not None:
            cfg[name] = value
    selected = cfg[cfg["setup"]]
    cfg = {k: v for k, v in cfg.items() if k not in {"simulation", "pub"}}
    cfg.update(selected)
    if cfg["model"] not in MODELS or cfg["setup"] not in {"simulation", "pub"}:
        raise ValueError("Unknown model or setup")
    if not (pd.Timestamp(cfg["test_end"]) < pd.Timestamp(cfg["train_start"])):
        raise ValueError("Test target period must precede the training target period")
    if not (0 < cfg["epochs"] <= cfg["epochs_scheduler"]) or not 0 <= cfg["monitor_fraction"] < 1:
        raise ValueError("Invalid training schedule")
    if cfg["static_scaling"] not in {"train", "test_fold"}:
        raise ValueError("static_scaling must be train or test_fold")
    for key in ("camels_root", "basin_file", "split_file", "output_root"):
        cfg[key] = str((path.resolve().parent / cfg[key]).resolve())
    return cfg


def build_model(cfg: dict):
    kw = dict(d_model=cfg["d_model"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
              dt_min=cfg["min_dt"], dt_max=cfg["max_dt"], modulation_scale=cfg["modulation_scale"],
              scan_mode=cfg["scan_mode"], positive_output=False, norm_type="batchnorm")
    if cfg["model"] in MODES:
        return SelectiveSSMConditioned(static_dim=cfg["static_dim"], readout_scale=cfg["readout_scale"],
                                      conditioning_mode=MODES[cfg["model"]], **kw)
    return SelectiveSSMInputOnly(d_input=32 if cfg["model"] == "concat_static" else 5, **kw)


def basin_split(cfg: dict) -> tuple[list[str], list[str]]:
    basins = Path(cfg["basin_file"]).read_text(encoding="utf-8-sig").split()
    if not basins or len(set(basins)) != len(basins) or any(len(b) != 8 or not b.isdigit() for b in basins):
        raise ValueError("Expected distinct eight-digit basin IDs")
    if cfg["setup"] == "simulation":
        return basins, basins
    folds = json.loads(Path(cfg["split_file"]).read_text(encoding="utf-8"))
    if set(folds) != {str(i) for i in range(12)}:
        raise ValueError("Expected twelve PUB folds")
    seen = []
    for fold in folds.values():
        train, test = fold["train"], fold["test"]
        if (len(set(train)) != len(train) or len(set(test)) != len(test)
                or set(train) & set(test) or set(train) | set(test) != set(basins)):
            raise ValueError("Invalid PUB basin partition")
        seen.extend(test)
    if len(seen) != len(basins) or set(seen) != set(basins):
        raise ValueError("Each basin must occur in exactly one test fold")
    fold = folds[str(cfg["split"])]
    return fold["train"], fold["test"]


def forward(model, batch, cfg):
    x, static, y, std = [v.to(cfg["device"]) for v in batch]
    if cfg["model"] in MODES:
        pred = model(x, static)
    elif cfg["model"] == "concat_static":
        pred = model(torch.cat([x, static.expand(-1, x.shape[1], -1)], dim=-1))
    else:
        pred = model(x)
    if pred.shape != y.shape or std.shape != y.shape:
        raise ValueError("Prediction, target and basin standard deviation must have matching shapes")
    return pred, y, std


def optimizer_for(model, cfg):
    tokens = ("bias", "norm", "raw_lambda", "B0", "C0", "D", "delta_proj", "gate_proj",
              "b_mod_proj", "c_mod_proj", "readout_mod")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or any(t in name for t in tokens) else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.5 * (1 + np.cos(np.pi * e / cfg["epochs_scheduler"])))
    return optimizer, scheduler


def train(cfg: dict, run: Path | None = None) -> Path:
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_basins, test_basins = basin_split(cfg)
    tag = f"{cfg['model']}_{cfg['setup']}_seed{cfg['seed']}"
    if cfg["setup"] == "pub":
        tag += f"_split{cfg['split']}"
    run = run or Path(cfg["output_root"]) / f"{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "config.json", cfg)
    save_json(run / "basins.json", {"train": train_basins, "test": test_basins})
    dataset = CamelsWindows(cfg, train_basins, train=True)
    save_json(run / "static_scaler.json", dataset.scaler)
    count = int(len(dataset) * cfg["monitor_fraction"])
    if count:
        training, monitoring = random_split(dataset, [len(dataset) - count, count],
                                           generator=torch.Generator().manual_seed(cfg["seed"]))
        monitor_loader = DataLoader(monitoring, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=0)
    else:
        training, monitor_loader = dataset, None
    loader = DataLoader(training, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    model = build_model(cfg).to(cfg["device"])
    optimizer, scheduler = optimizer_for(model, cfg)
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        iterator = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        lr = optimizer.param_groups[0]["lr"]
        for batch in iterator:
            optimizer.zero_grad()
            pred, y, std = forward(model, batch, cfg)
            loss = ((pred - y).square() * (1 / (std + cfg["loss_eps"]).square())).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            iterator.set_postfix(loss=f"{losses[-1]:.5f}")
        monitored = []
        model.eval()
        if monitor_loader is not None:
            with torch.no_grad():
                for batch in monitor_loader:
                    pred, y, std = forward(model, batch, cfg)
                    value = ((pred - y).square() * (1 / (std + cfg["loss_eps"]).square())).mean()
                    if not torch.isfinite(value):
                        raise FloatingPointError("Non-finite monitoring loss")
                    monitored.append(value.item())
        history.append({"epoch": epoch, "lr": lr, "train_loss": np.mean(losses),
                        "monitor_loss": np.mean(monitored) if monitored else np.nan})
        pd.DataFrame(history).to_csv(run / "loss.csv", index=False)
        torch.save(model.state_dict(), run / f"epoch{epoch}.pt")
        scheduler.step()
    print(f"Run: {run.resolve()}")
    return run


def evaluate(run: Path) -> Path:
    if version("neuralhydrology") != "1.13.0":
        raise RuntimeError("Install neuralhydrology==1.13.0")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    basins = json.loads((run / "basins.json").read_text(encoding="utf-8"))["test"]
    scaler = json.loads((run / "static_scaler.json").read_text(encoding="utf-8"))
    if cfg["static_scaling"] == "test_fold":
        scaler = static_scaler(attributes(Path(cfg["camels_root"]), basins))
    model = build_model(cfg).to(cfg["device"])
    model.load_state_dict(torch.load(run / f"epoch{cfg['epochs']}.pt", map_location=cfg["device"], weights_only=True))
    model.eval()
    out = run / f"evaluation_epoch{cfg['epochs']}"
    out.mkdir(exist_ok=False)
    (out / "predictions").mkdir()
    save_json(out / "static_scaler.json", scaler)
    rows = []
    for basin in tqdm(basins, desc="Evaluate"):
        dataset = CamelsWindows(cfg, [basin], train=False, scaler=scaler)
        predictions, observations = [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=0):
                pred, y, _ = forward(model, batch, cfg)
                predictions.append(pred.cpu().numpy())
                observations.append(y.cpu().numpy())
        sim = np.maximum(np.concatenate(predictions).ravel().astype(np.float64) * OUTPUT_STD + OUTPUT_MEAN, 0)
        obs = np.concatenate(observations).ravel().astype(np.float64)
        if not np.isfinite(sim).all():
            raise FloatingPointError("Non-finite discharge prediction")
        frame = pd.DataFrame({"qobs": obs, "qsim": sim}, index=dataset.series[0]["dates"])
        frame.to_csv(out / "predictions" / f"{basin}.csv", index_label="date")
        metrics = {m: np.nan for m in METRICS}
        if np.isfinite(obs).any():
            metrics = calculate_metrics(xr.DataArray(obs, dims=["date"]),
                                        xr.DataArray(sim, dims=["date"]), metrics=METRICS)
        rows.append({"basin": basin, "n_valid": int(np.isfinite(obs).sum()), **metrics})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    finite = scores[METRICS].replace([np.inf, -np.inf], np.nan)
    summary = finite.agg(["mean", "median", "std", "count"]).T
    summary["n_nan"] = scores[METRICS].isna().sum()
    summary["n_inf"] = np.isinf(scores[METRICS]).sum()
    summary.to_csv(out / "summary.csv", index_label="metric")
    print(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--setup", choices=["simulation", "pub"])
    parser.add_argument("--split", type=int, choices=range(12))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(load_config(args.config, args.model, args.setup, args.split), args.run_dir)
    elif args.run_dir is None:
        parser.error("evaluate requires --run-dir")
    else:
        evaluate(args.run_dir)


if __name__ == "__main__":
    main()
