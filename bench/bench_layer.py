"""Training-speed benchmark for GRULayer.

Measures forward-only and forward+backward latency for our GRULayer against
torch.nn.GRU (cuDNN) as the ceiling reference. Variants are registered in
VARIANTS so later optimization steps can plug in without touching the
harness.

Run:
    uv run python bench/bench_layer.py
    uv run python bench/bench_layer.py --shapes 32,16,256 64,32,512
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS

Shape = tuple[int, int, int, int]  # (seq, batch, in, hidden)
LayerFn = Callable[[Shape], nn.Module]


# ---------------------------------------------------------------------------
# Variants — each builds an nn.Module with signature forward(x) or
# forward(x, h0). The harness handles the calling convention.
# ---------------------------------------------------------------------------


def build_cudnn_gru(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return nn.GRU(in_dim, hid, num_layers=1, batch_first=False)


def build_ours_fp32(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(in_dim, hid, recipe=PRESETS["fp32"])


def build_ours_int8(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(in_dim, hid, recipe=PRESETS["int8_per_channel"])


def build_ours_int8_fused(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(
        in_dim, hid, recipe=PRESETS["int8_per_channel"], gate_layout="fused"
    )


def build_ours_int8_fused_compiled(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(
        in_dim,
        hid,
        recipe=PRESETS["int8_per_channel"],
        gate_layout="fused",
        compile_step=True,
    )


def build_ours_int8_prebatch(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(
        in_dim,
        hid,
        recipe=PRESETS["int8_per_channel"],
        gate_layout="fused",
        pre_batch_input=True,
    )


def build_ours_int8_prebatch_compiled(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    return GRULayer(
        in_dim,
        hid,
        recipe=PRESETS["int8_per_channel"],
        gate_layout="fused",
        pre_batch_input=True,
        compile_step=True,
    )


def build_ours_fp32_fused_compiled(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    from gru_qat.quantizers import QuantizerConfig, QuantRecipe

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    return GRULayer(in_dim, hid, recipe=rec, gate_layout="fused", compile_step=True)


def build_ours_fp32_fused(shape: Shape) -> nn.Module:
    _, _, in_dim, hid = shape
    # PRESETS["fp32"] has axis=None (Identity) — substitute axis=0 to satisfy
    # the fused-gate guard. Identity is still no-op so behaviour is unchanged.
    from gru_qat.quantizers import QuantizerConfig, QuantRecipe

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    return GRULayer(in_dim, hid, recipe=rec, gate_layout="fused")


VARIANTS: dict[str, LayerFn] = {
    "cudnn_gru_fp32": build_cudnn_gru,
    "ours_fp32_identity": build_ours_fp32,
    "ours_int8_per_channel": build_ours_int8,
    "ours_int8_fused": build_ours_int8_fused,
    "ours_int8_fused_compiled": build_ours_int8_fused_compiled,
    "ours_int8_prebatch": build_ours_int8_prebatch,
    "ours_int8_prebatch_compiled": build_ours_int8_prebatch_compiled,
}

BF16_VARIANTS: set[str] = set()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    name: str
    shape: Shape
    train_ms: float  # forward + backward + opt.zero_grad


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_callable(fn: Callable[[], None], n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        fn()
    _sync()
    samples: list[float] = []
    for _ in range(n_iter):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def bench_one(
    name: str,
    layer_fn: LayerFn,
    shape: Shape,
    n_warmup: int,
    n_iter: int,
    *,
    autocast_bf16: bool = False,
) -> TimingResult:
    seq, batch, in_dim, hid = shape
    device = torch.device("cuda")
    model = layer_fn(shape).to(device)
    model.train()

    x = torch.randn(seq, batch, in_dim, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        if autocast_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
        else:
            out = model(x)
        y = out[0] if isinstance(out, tuple) else out
        loss = y.float().pow(2).mean()
        loss.backward()
        # Skip opt.step — it's not on the GRU hot path and adds noise.

    trn = _time_callable(train_step, n_warmup, n_iter)
    return TimingResult(name, shape, trn)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_shape(s: str) -> Shape:
    parts = [int(p) for p in s.split(",")]
    if len(parts) == 3:
        seq, batch, hid = parts
        return (seq, batch, hid, hid)
    if len(parts) == 4:
        return (parts[0], parts[1], parts[2], parts[3])
    raise ValueError(f"shape must be seq,batch,hidden or seq,batch,in,hidden: {s}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["32,16,256", "64,32,512"],
        help="seq,batch,hidden  or  seq,batch,in,hidden",
    )
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iter", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — bench is GPU-only.")

    # TF32: on Ampere+ this routes fp32 matmuls through tensor cores at
    # ~bf16-mantissa precision. ~2x fp32 GEMM speedup with negligible
    # impact on QAT — the fake-quant noise dominates the TF32 noise.
    torch.set_float32_matmul_precision("high")

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# torch:  {torch.__version__}")
    print(f"# tf32:   high (set_float32_matmul_precision)")
    print(f"# warmup={args.warmup} iter={args.iter}")
    print()
    header = f"{'variant':32s} {'shape':22s} {'train ms':>10s}  {'vs cudnn':>10s}"
    print(header)
    print("-" * len(header))

    for shape_str in args.shapes:
        shape = parse_shape(shape_str)
        ref_trn: float | None = None
        ref_ours: float | None = None  # baseline ours_int8_per_channel

        for name in args.variants:
            torch.manual_seed(0)
            r = bench_one(
                name,
                VARIANTS[name],
                shape,
                args.warmup,
                args.iter,
                autocast_bf16=name in BF16_VARIANTS,
            )
            shape_str_fmt = f"({r.shape[0]},{r.shape[1]},{r.shape[2]},{r.shape[3]})"
            ratio = (
                f"{r.train_ms / ref_trn:5.2f}x" if ref_trn is not None else "      —"
            )
            speedup = (
                f" speedup={ref_ours / r.train_ms:4.2f}x"
                if ref_ours is not None
                else ""
            )
            line = (
                f"{r.name:32s} {shape_str_fmt:22s} "
                f"{r.train_ms:10.3f}  {ratio:>10s}{speedup}"
            )
            print(line)
            if "cudnn" in name and ref_trn is None:
                ref_trn = r.train_ms
            if name == "ours_int8_per_channel" and ref_ours is None:
                ref_ours = r.train_ms
        print()


if __name__ == "__main__":
    main()
