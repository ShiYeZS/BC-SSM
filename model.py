"""Basin-conditioned selective SSM for rainfall-runoff simulation.

Separate encoders represent daily forcing and static catchment attributes.
The basin embedding conditions selective state dynamics and the pooled FiLM
readout. Positive decay rates and bounded parameter modulation define the
diagonal recurrence. Five configurations expose the static-input, state,
and readout pathways within the same sequence-to-one framework.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SELECTIVE_CONDITIONED_MODES = {
    "selective_ssm": "both",
    "selective_ssm_conditioned": "both",
    "selective_ssm_state_only": "state_only",
    "selective_ssm_readout_only": "readout_only",
}


class SelectiveSSMBlock(nn.Module):
    """Stable diagonal selective SSM block.

    The transition is diagonal and stable by construction:
    state_t = exp(-dt_t * lambda) * state_{t-1}
              + (1 - exp(-dt_t * lambda)) * B_t * x_t

    Current sequence features generate B_t, C_t, dt_t, and the output gate.
    When enabled, the basin embedding jointly conditions these parameters.
    B_t and C_t use bounded multiplicative modulation of learned base vectors.
    """

    def __init__(
        self,
        d_model: int,
        static_dim: int,
        dropout: float,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
        modulation_scale: float = 0.1,
        prenorm: bool = False,
        scan_mode: str = "parallel",
        norm_type: str = "batchnorm",
    ) -> None:
        super().__init__()
        if dt_min <= 0:
            raise ValueError(f"dt_min must be positive, got {dt_min}")
        if dt_max <= dt_min:
            raise ValueError(f"dt_max must be greater than dt_min, got {dt_max} <= {dt_min}")
        if modulation_scale < 0:
            raise ValueError(f"modulation_scale must be non-negative, got {modulation_scale}")
        if static_dim < 0:
            raise ValueError(f"static_dim must be non-negative, got {static_dim}")
        if scan_mode not in {"parallel", "loop"}:
            raise ValueError(f"scan_mode must be 'parallel' or 'loop', got {scan_mode}")
        if norm_type not in {"batchnorm", "layernorm"}:
            raise ValueError(f"norm_type must be 'batchnorm' or 'layernorm', got {norm_type}")

        self.d_model = d_model
        self.static_dim = static_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.modulation_scale = modulation_scale
        self.prenorm = prenorm
        self.scan_mode = scan_mode
        self.norm_type = norm_type

        cond_dim = d_model + static_dim
        self.delta_proj = nn.Linear(cond_dim, d_model)
        self.gate_proj = nn.Linear(cond_dim, d_model)
        self.b_mod_proj = nn.Linear(cond_dim, d_model)
        self.c_mod_proj = nn.Linear(cond_dim, d_model)

        init_lambda = torch.linspace(0.1, 1.0, d_model)
        self.raw_lambda = nn.Parameter(torch.log(torch.expm1(init_lambda)))
        self.B0 = nn.Parameter(torch.ones(d_model))
        self.C0 = nn.Parameter(torch.ones(d_model))
        self.D = nn.Parameter(torch.zeros(d_model))

        self.output_linear = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GLU(dim=-1),
        )
        self.dropout = nn.Dropout1d(dropout) if norm_type == "batchnorm" else nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model) if norm_type == "batchnorm" else nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        basin_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"Expected x with shape [B, T, D], got {tuple(x.shape)}")
        if self.static_dim > 0:
            if basin_embedding is None:
                raise RuntimeError("This block requires a basin embedding")
            if basin_embedding.ndim != 2 or basin_embedding.shape[-1] != self.static_dim:
                raise RuntimeError(
                    f"Expected basin_embedding [B, {self.static_dim}], "
                    f"got {tuple(basin_embedding.shape)}"
                )
            if x.shape[0] != basin_embedding.shape[0]:
                raise RuntimeError("Batch size mismatch between sequence and basin embedding")
        elif basin_embedding is not None:
            raise RuntimeError("Input-only selective SSM block does not accept a basin embedding")

        residual = x
        z = self._apply_norm(x) if self.prenorm else x
        y = self._selective_scan(z, basin_embedding)
        y = self.output_linear(y)
        y = self._apply_dropout(y)
        out = residual + y
        out = out if self.prenorm else self._apply_norm(out)
        return out

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "batchnorm":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "batchnorm":
            return self.dropout(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(x)

    def _selective_scan(
        self,
        x: torch.Tensor,
        basin_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise RuntimeError(f"Expected d_model={self.d_model}, got {d_model}")

        if basin_embedding is None:
            eta = x
        else:
            basin_seq = basin_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            eta = torch.cat([x, basin_seq], dim=-1)

        delta = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.delta_proj(eta))
        gate = torch.sigmoid(self.gate_proj(eta))
        b_t = self.B0.view(1, 1, -1) * (
            1.0 + self.modulation_scale * torch.tanh(self.b_mod_proj(eta))
        )
        c_t = self.C0.view(1, 1, -1) * (
            1.0 + self.modulation_scale * torch.tanh(self.c_mod_proj(eta))
        )

        decay_rate = F.softplus(self.raw_lambda).view(1, 1, -1) + 1e-4
        decay = torch.exp(-delta * decay_rate)
        drive = (1.0 - decay) * b_t * x

        if self.scan_mode == "loop":
            state = self._scan_recurrence_loop(decay, drive)
        else:
            state = self._scan_recurrence_parallel(decay, drive)

        emission = gate * (c_t * state + self.D.view(1, 1, -1) * x)
        return emission

    @staticmethod
    def _scan_recurrence_loop(decay: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Sequential scan of s_t = decay_t * s_{t-1} + drive_t."""
        if decay.shape != drive.shape:
            raise RuntimeError(f"decay/drive shape mismatch: {tuple(decay.shape)} vs {tuple(drive.shape)}")
        batch_size, _, d_model = decay.shape
        state = decay.new_zeros(batch_size, d_model)
        states = []
        for t in range(decay.shape[1]):
            state = decay[:, t, :] * state + drive[:, t, :]
            states.append(state)
        return torch.stack(states, dim=1)

    @staticmethod
    def _scan_recurrence_parallel(decay: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Parallel inclusive scan for diagonal affine recurrences.

        Each time step is an affine map f_t(s) = decay_t * s + drive_t.
        The associative composition
        (a2, b2) o (a1, b1) = (a2 * a1, b2 + a2 * b1)
        computes all prefix states from a zero initial state in O(log T)
        parallel stages. Floating-point sums follow the composition order.
        """
        if decay.shape != drive.shape:
            raise RuntimeError(f"decay/drive shape mismatch: {tuple(decay.shape)} vs {tuple(drive.shape)}")
        if decay.ndim != 3:
            raise RuntimeError(f"Expected [B, T, D] tensors, got {tuple(decay.shape)}")

        a = decay
        b = drive
        step = 1
        seq_len = decay.shape[1]
        while step < seq_len:
            a_prev = a[:, :-step, :]
            b_prev = b[:, :-step, :]
            a_cur = a[:, step:, :]
            b_cur = b[:, step:, :]
            a = torch.cat([a[:, :step, :], a_cur * a_prev], dim=1)
            b = torch.cat([b[:, :step, :], b_cur + a_cur * b_prev], dim=1)
            step *= 2
        return b


class SelectiveSSMConditioned(nn.Module):
    """BC-SSM with independently configurable basin-conditioning pathways.

    ``both`` conditions selective dynamics and the FiLM readout;
    ``state_only`` conditions dynamics and uses a linear readout;
    ``readout_only`` applies basin conditioning exclusively to the readout.
    Every mode retains input-dependent selective dynamics.
    """

    def __init__(
        self,
        d_input: int = 5,
        d_static: int = 27,
        d_output: int = 1,
        d_model: int = 128,
        static_dim: int = 64,
        n_layers: int = 6,
        dropout: float = 0.12,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
        modulation_scale: float = 0.1,
        readout_scale: float = 0.1,
        prenorm: bool = False,
        positive_output: bool = False,
        scan_mode: str = "parallel",
        norm_type: str = "batchnorm",
        conditioning_mode: str = "both",
    ) -> None:
        super().__init__()
        if d_input != 5:
            raise ValueError("SelectiveSSMConditioned expects separated 5-variable dynamic forcing input")
        if d_static <= 0:
            raise ValueError(f"d_static must be positive, got {d_static}")
        if static_dim <= 0:
            raise ValueError(f"static_dim must be positive, got {static_dim}")
        if conditioning_mode not in {"both", "state_only", "readout_only"}:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")

        self.d_input = d_input
        self.d_static = d_static
        self.conditioning_mode = conditioning_mode
        self.condition_state = conditioning_mode != "readout_only"
        self.condition_readout = conditioning_mode != "state_only"
        self.positive_output = positive_output
        self.readout_scale = readout_scale
        if scan_mode not in {"parallel", "loop"}:
            raise ValueError(f"scan_mode must be 'parallel' or 'loop', got {scan_mode}")
        if norm_type not in {"batchnorm", "layernorm"}:
            raise ValueError(f"norm_type must be 'batchnorm' or 'layernorm', got {norm_type}")
        self.scan_mode = scan_mode
        self.norm_type = norm_type

        self.dynamic_encoder = nn.Linear(d_input, d_model)
        self.static_encoder = nn.Sequential(
            nn.Linear(d_static, static_dim),
            nn.GELU(),
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, static_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=d_model,
                    static_dim=static_dim if self.condition_state else 0,
                    dropout=dropout,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    modulation_scale=modulation_scale,
                    prenorm=prenorm,
                    scan_mode=scan_mode,
                    norm_type=norm_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.readout_mod = nn.Linear(static_dim, 2 * d_model) if self.condition_readout else None
        self.decoder = nn.Linear(d_model, d_output)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        if dynamic.ndim != 3:
            raise RuntimeError(f"Expected dynamic input [B, T, 5], got {tuple(dynamic.shape)}")
        if dynamic.shape[-1] != self.d_input:
            raise RuntimeError(f"Expected {self.d_input} dynamic features, got {dynamic.shape[-1]}")

        static = self._prepare_static(static)
        if static.shape[0] != dynamic.shape[0]:
            raise RuntimeError("Batch size mismatch between dynamic and static inputs")
        z_b = self.static_encoder(static)
        h = self.dynamic_encoder(dynamic)

        state_condition = z_b if self.condition_state else None
        for layer in self.layers:
            h = layer(h, state_condition)

        pooled = h.mean(dim=1)
        conditioned = pooled
        if self.readout_mod is not None:
            gamma_raw, beta_raw = self.readout_mod(z_b).chunk(2, dim=-1)
            gamma = 1.0 + self.readout_scale * torch.tanh(gamma_raw)
            beta = self.readout_scale * torch.tanh(beta_raw)
            conditioned = gamma * pooled + beta
        out = self.decoder(conditioned)

        # Standardized discharge uses the default linear output.
        out = F.softplus(out) if self.positive_output else out
        return out

    def _prepare_static(self, static: torch.Tensor) -> torch.Tensor:
        if static.ndim == 3:
            if static.shape[1] != 1:
                raise RuntimeError(f"Expected static [B, 1, {self.d_static}], got {tuple(static.shape)}")
            static = static.squeeze(1)
        if static.ndim != 2:
            raise RuntimeError(f"Expected static [B, {self.d_static}], got {tuple(static.shape)}")
        if static.shape[-1] != self.d_static:
            raise RuntimeError(f"Expected {self.d_static} static features, got {static.shape[-1]}")
        return static


class SelectiveSSMInputOnly(nn.Module):
    """BC-SSM input-path configurations with a pooled linear readout.

    ``concat_static`` encodes five forcing variables and 27 static attributes
    at each time step; ``no_static`` encodes the five forcing variables.
    Both generate selective parameters from the encoded sequence directly.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int = 1,
        d_model: int = 128,
        n_layers: int = 6,
        dropout: float = 0.12,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
        modulation_scale: float = 0.1,
        prenorm: bool = False,
        positive_output: bool = False,
        scan_mode: str = "parallel",
        norm_type: str = "batchnorm",
    ) -> None:
        super().__init__()
        if d_input <= 0:
            raise ValueError(f"d_input must be positive, got {d_input}")

        self.d_input = d_input
        self.positive_output = positive_output
        self.input_encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=d_model,
                    static_dim=0,
                    dropout=dropout,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    modulation_scale=modulation_scale,
                    prenorm=prenorm,
                    scan_mode=scan_mode,
                    norm_type=norm_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise RuntimeError(f"Expected input [B, T, D], got {tuple(inputs.shape)}")
        if inputs.shape[-1] != self.d_input:
            raise RuntimeError(f"Expected {self.d_input} input features, got {inputs.shape[-1]}")

        h = self.input_encoder(inputs)
        for layer in self.layers:
            h = layer(h)

        out = self.decoder(h.mean(dim=1))
        return F.softplus(out) if self.positive_output else out
