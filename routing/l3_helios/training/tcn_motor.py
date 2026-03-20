"""
routing/l3_helios/training/tcn_motor.py

Skeleton TCN for L1 motor sensor stream.
Train on RTX 4070 (CUDA). Target: sub-20ms inference.

TCN is preferred over LTFAD for this stream due to multi-scale
temporal patterns in motor RPM/torque. LTFAD is documented in
brainstorm/model_decisions/ as evaluated alternative.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution. No future leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, time)
        h = self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h).transpose(1, 2)).transpose(1, 2)
        return self.act(h + self.residual(x))


class MotorTCN(nn.Module):
    """
    L1 motor sensor expert.
    Input: (batch, seq_len, n_features)
    Output: (batch, seq_len, n_features)  -- next-step prediction per feature
    Also outputs anomaly_score: (batch,) -- scalar reconstruction error proxy

    Features expected (in order):
        motor_rpm, motor_torque_nm, motor_temp_c, motor_current_a
    """

    N_FEATURES = 4
    CHANNELS = [32, 64, 64, 32]
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8]

    def __init__(self):
        super().__init__()
        layers = []
        in_ch = self.N_FEATURES
        for out_ch, dil in zip(self.CHANNELS, self.DILATIONS):
            layers.append(TCNBlock(in_ch, out_ch, self.KERNEL_SIZE, dil))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(self.CHANNELS[-1], self.N_FEATURES)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        h = x.transpose(1, 2)                      # (batch, n_features, seq_len)
        h = self.encoder(h)                         # (batch, channels, seq_len)
        h = h.transpose(1, 2)                       # (batch, seq_len, channels)
        pred = self.head(h)                         # (batch, seq_len, n_features)
        recon_error = ((pred - x) ** 2).mean(dim=(1, 2))   # (batch,)
        return pred, recon_error

    def inference_latency_check(self, seq_len=100, device="cuda"):
        """Quick latency check. Run after AWQ quantization."""
        import time
        self.eval()
        x = torch.randn(1, seq_len, self.N_FEATURES).to(device)
        self.to(device)
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                self(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                self(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / 100 * 1000
        print(f"TCN motor inference: {elapsed:.2f}ms (target: <20ms)")
        return elapsed


if __name__ == "__main__":
    model = MotorTCN()
    x = torch.randn(4, 100, MotorTCN.N_FEATURES)
    pred, err = model(x)
    print(f"pred: {pred.shape}, recon_err: {err.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
