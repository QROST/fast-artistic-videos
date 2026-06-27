"""Throughput benchmark for the generator (forward + backward).

Times the configured model at the chosen precision / compile setting so the
Phase-2 throughput knobs can be measured on the target hardware (e.g. M5 Max).
"""

from __future__ import annotations

import time

import torch

from fav.device import autocast_context, select_device, synchronize
from fav.models.generator import IN_CHANNELS, build_model


def benchmark(cfg, device=None, size: int = 256, iters: int = 20, warmup: int = 5) -> dict:
    device = select_device(device or cfg.device)
    model = build_model(cfg.model).to(device)
    if getattr(cfg, "compile_model", False):
        model = torch.compile(model)
    precision = getattr(cfg, "precision", "fp32")
    x = torch.randn(cfg.batch_size, IN_CHANNELS, size, size, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            out = model(x)
            loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    return {
        "device": str(device),
        "precision": precision,
        "compile": bool(getattr(cfg, "compile_model", False)),
        "grad_checkpoint": bool(getattr(cfg, "grad_checkpoint", False)),
        "batch": cfg.batch_size,
        "size": size,
        "ms_per_iter": round(dt * 1000, 2),
        "iter_per_s": round(1.0 / dt, 2),
    }
