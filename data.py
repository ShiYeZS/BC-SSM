"""CAMELS inputs for BC-SSM training and evaluation.

Read NLDAS-extended forcing, catchment attributes, and daily discharge;
construct sequence windows on demand. Source attribution: licenses/NOTICE.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

INPUT_MEAN = np.array([3.015, 357.68, 10.864, 10.864, 1055.533])
INPUT_STD = np.array([7.573, 129.878, 10.932, 10.932, 705.998])
OUTPUT_MEAN, OUTPUT_STD = 1.49996196, 3.62443672
FORCING = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
EXCLUDE = {
    "gauge_name", "area_geospa_fabric", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "dom_land_cover_frac", "dom_land_cover",
    "high_prec_timing", "low_prec_timing", "huc", "huc_02", "gauge_lat", "gauge_lon",
    "q_mean", "runoff_ratio", "stream_elas", "slope_fdc", "baseflow_index", "hfd_mean",
    "q5", "q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq",
    "geol_porostiy", "root_depth_50", "root_depth_99", "organic_frac", "water_frac", "other_frac",
}


def attributes(root: Path, basins: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
              for p in (root / "camels_attributes_v2.0").glob("camels_*.txt")]
    frame = pd.concat(frames, axis=1)
    frame = frame.loc[frame.index.isin(basins)]
    frame = frame.drop(columns=[c for c in frame.columns if c in EXCLUDE])
    if frame.shape != (len(basins), 27) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Expected 27 finite attributes for every basin")
    return frame


def static_scaler(frame: pd.DataFrame) -> dict:
    mean, std = frame.mean(), frame.std()
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Static scaling requires nonzero attribute standard deviations")
    return {"features": frame.columns.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def read_daily(root: Path, basin: str, start: str, end: str, length: int) -> pd.DataFrame:
    path = find_file(root / "basin_mean_forcing/nldas_extended", f"{basin}*_forcing_leap.txt")
    with path.open() as stream:
        stream.readline()
        stream.readline()
        area = int(stream.readline())
    frame = pd.read_csv(path, sep=r"\s+", header=3)
    frame.index = pd.to_datetime(dict(year=frame.Year, month=frame.Mnth, day=frame.Day))
    frame = frame.rename(columns=dict(zip(
        ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"], FORCING)))
    q = pd.read_csv(find_file(root / "usgs_streamflow", f"{basin}*_streamflow_qc.txt"),
                    sep=r"\s+", header=None, names=["basin", "year", "month", "day", "q", "flag"])
    q.index = pd.to_datetime(q[["year", "month", "day"]])
    frame["qobs"] = 28316846.592 * q.q * 86400 / (area * 10**6)
    dates = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=length - 1), end)
    frame = frame.loc[dates[0]:dates[-1]]
    if not frame.index.equals(dates) or not np.isfinite(frame[FORCING].to_numpy()).all():
        raise ValueError(f"Incomplete or non-finite forcing: {basin}")
    return frame


class CamelsWindows(Dataset):
    """Store each daily series once and slice windows during batch loading."""

    def __init__(self, cfg: dict, basins: list[str], train: bool, scaler: dict | None = None):
        self.length = cfg["seq_length"]
        root = Path(cfg["camels_root"])
        attrs = attributes(root, basins)
        self.scaler = static_scaler(attrs) if scaler is None else scaler
        attrs = (attrs[self.scaler["features"]] - self.scaler["mean"]) / self.scaler["std"]
        self.series, counts = [], []
        start, end = (cfg["train_start"], cfg["train_end"]) if train else (cfg["test_start"], cfg["test_end"])
        for basin in tqdm(basins, desc="Load CAMELS"):
            frame = read_daily(root, basin, start, end, self.length)
            x = ((frame[FORCING].to_numpy() - INPUT_MEAN) / INPUT_STD).astype(np.float32)
            y = frame.qobs.to_numpy()[self.length - 1:].copy()
            valid = np.isfinite(y) & (y >= 0)
            if train and not valid.any():
                raise ValueError(f"No valid training observations: {basin}")
            q_std = np.float32(np.std(y[valid])) if valid.any() else np.float32(np.nan)
            indices = np.flatnonzero(valid) if train else np.arange(len(y))
            y[~valid] = np.nan
            if train:
                y = (y - OUTPUT_MEAN) / OUTPUT_STD
            self.series.append({"basin": basin, "x": x, "y": y.astype(np.float32),
                                "indices": indices, "dates": frame.index[self.length - 1:][indices],
                                "static": attrs.loc[basin].to_numpy(dtype=np.float32)[None, :],
                                "std": np.array([q_std], dtype=np.float32)})
            counts.append(len(indices))
        self.ends = np.cumsum(counts)

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int):
        basin_index = int(np.searchsorted(self.ends, index, side="right"))
        offset = int(self.ends[basin_index - 1]) if basin_index else 0
        item = self.series[basin_index]
        start = item["indices"][index - offset]
        return (torch.from_numpy(item["x"][start:start + self.length]),
                torch.from_numpy(item["static"]), torch.from_numpy(item["y"][start:start + 1]),
                torch.from_numpy(item["std"]))
