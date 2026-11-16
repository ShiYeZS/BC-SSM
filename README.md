# BC-SSM

**Basin-Conditioned Selective State Space Model for Daily Rainfall-Runoff Simulation**

BC-SSM represents catchment attributes as a basin embedding that jointly
conditions selective state dynamics and the final discharge readout.

- **Basin-conditioned dynamics:** daily features and the basin embedding
  jointly determine the update step, input modulation, state emission, and gate.
- **Stable parameterization:** positive decay rates and bounded modulation
  govern the diagonal state recurrence, computed with a parallel scan.
- **Conditioned readout:** basin-dependent FiLM transforms the temporally
  pooled representation before the scalar discharge prediction.

## Configurations

| `model` | Static information pathway |
| --- | --- |
| `bc_ssm` | Basin embedding conditions both dynamics and FiLM readout. |
| `no_static` | Dynamic forcing only, with a pooled linear readout. |
| `concat_static` | Static attributes concatenated with daily forcing. |
| `state_only` | Basin embedding conditions dynamics; linear readout. |
| `readout_only` | Basin embedding conditions only the FiLM readout. |

## Run

All experiment settings are in `config.json`. Set `camels_root` to the CAMELS
directory containing NLDAS-extended forcing. Paths are relative to the config.

```console
python -m pip install -r requirements.txt
python main.py train --model bc_ssm
python main.py train --model no_static
python main.py train --model concat_static
python main.py train --model state_only
python main.py train --model readout_only
python main.py train --setup pub --split 0
python main.py evaluate --run-dir runs/RUN_NAME
```

`model.py` defines the five configurations; `data.py` loads CAMELS inputs;
`main.py` trains and evaluates simulation and PUB experiments. PUB split
numbers are 0-11. `static_scaling` selects training-basin (`train`) or
evaluation-fold (`test_fold`) attribute statistics at inference.

Evaluation exports daily predictions, basin-wise metrics, and summary CSVs.
Discharge is converted to mm/day and clipped at zero. Metrics are computed
directly with NeuralHydrology 1.13.0. Attribution and licenses are in `licenses/`.
