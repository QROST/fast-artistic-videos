# Phase 3 — diffusion / per-style LoRA roadmap

Kickoff for Phase 3, after **Phase 1** (faithful PyTorch/MPS port) and **Phase 2**
(SOTA module upgrades) are merged. GitHub Issues are disabled on this repo, so
this doc is the discussion artifact.

## Where we are (the baseline)

A faithful, reproducible, MPS-ready **feed-forward** video style-transfer system:
per-style network, explicit optical-flow temporal consistency, occlusion
validated **bit-for-bit against the original C++ `consistencyChecker`**. Phase 2
added (all config-gated, defaults unchanged): bf16/`torch.compile` throughput +
`fav bench`, SEA-RAFT/FlowFormer flow backends, optional LPIPS/DINO perceptual
terms, GroupNorm + squeeze-excitation generator primitives.

- **Feed-forward strengths:** deterministic, near-real-time, temporally stable,
  high per-style quality.
- **Limits:** one network per style; style space bounded by the training style image.

## Decided direction

- **Quality first.**
- **Keep the per-style paradigm.**
- **Reuse the existing Phase 1/2 infrastructure; iterate gradually.**

→ **Per-style LoRA on a diffusion base + ControlNet conditioning driven by our
existing optical flow / occlusion**, with distillation to a fast feed-forward
student as a later option.

## Phased plan

| Step | What | Notes |
|---|---|---|
| **3a — reuse bridge** | `fav/diffusion/conditioning.py`: turn our `(dy,dx)` flow + reliability + warp into ControlNet-style conditioning tensors (warped-previous latent / flow / structure) | Pure tensor logic; reuses `fav.warp`/`fav.flow`/`fav.occlusion`; **unit-testable without `diffusers`** — the natural first brick |
| **3b — ControlNet + per-style LoRA** | SDXL/SD base + flow/structure ControlNet + one LoRA per style (tens of MB, ~minutes to train); per frame condition on the warped previous frame's latent + flow + occlusion | Quality core; inherits Phase-1's temporal mechanism; not yet real-time |
| **3c — temporal/cross-frame attention** | Share attention across adjacent frames | Pushes consistency beyond explicit flow |
| **3d — distillation (optional)** | Use 3b/3c as teacher → fast feed-forward student | Diffusion quality at Phase-1 speed; best fit for the M5 Max real-time goal |
| **side — learned occlusion** | Small net predicting occlusion/uncertainty (moved from PR-F) | Co-trainable with the above |

## Reuse points (Phase 1/2 → Phase 3)

`fav.warp` (grid_sample warp, flow I/O) · `fav.flow` (SEA-RAFT/RAFT) ·
`fav.occlusion` (golden-validated reliability) · `fav.preprocess` · `fav.data`
(real-video + synthetic sources) · `fav.losses` (LPIPS/DINO already wired). The
per-frame "warp previous output + mask by occlusion" recipe carries straight
over as diffusion conditioning.

## Environment note

Real Phase-3 work needs `diffusers` + a base-model download + MPS/GPU, so it
runs on the **M5 Max** (or a CUDA box), not the ephemeral CPU container used for
the faithful port. The feed-forward Phase 1/2 stays the fast, reproducible
default; diffusion is the **additive** quality path.

## Open questions

1. Base model: SDXL (quality) vs SD1.5 (lighter on MPS)?
2. Style conditioning: ControlNet vs IP-Adapter vs both?
3. Start 3a now (the testable reuse bridge), or wait until on the M5 Max for the
   full 3b prototype?
