# `fav` — Fast Artistic Videos, PyTorch/MPS port

`fav/` is a faithful PyTorch re-implementation of this repository's Torch7/Lua
method, runnable on CPU, CUDA, and **Apple-Silicon MPS** for both training and
inference. Phase 1 preserves the original algorithm and constants; only dead
native dependencies are modernized (optical flow → RAFT/SEA-RAFT, warping →
`grid_sample`, the C++ occlusion checker → vectorized torch, the VGG-16 loss net
→ converted weights, HDF5 → streaming datasets). The legacy `.lua` sources remain
as the reference.

## Install

```bash
pip install -e .            # core (torch, numpy)
pip install -e '.[flow,io,vr,convert,dev]'   # RAFT, image I/O, 360, t7 convert, tests
```

MPS is selected automatically (`fav.device.select_device`); `PYTORCH_ENABLE_MPS_FALLBACK=1`
is set so any unimplemented op falls back to CPU. Phase 1 runs in fp32.

## One-time: convert the VGG-16 loss network

Style appearance depends on the exact caffe VGG-16 features, so convert the
original `vgg16.t7` once (the 6 sample *style* models are not needed):

```bash
fav convert-vgg models/vgg16.t7 models/vgg16_caffe.pt
```

## Compute optical flow + occlusions for a clip

```bash
fav compute-flow --frames frames/ --out flow/ --backend raft --pattern 'frame_%05d.png'
```

Writes `backward_{cur}_{prev}.flo` and `reliable_{cur}_{prev}.pgm`, byte-compatible
with the legacy tooling.

### Flow backends (Phase 2)

The estimator is swappable via `--backend` (or `data.flow_backend` in training):

| backend | needs | notes |
|---|---|---|
| `raft` (default) | `torchvision` | out-of-box; `--model raft_large`/`raft_small` |
| `sea_raft` (**recommended**) | `pip install '.[ptlflow]'` | SOTA accuracy → better occlusion/consistency |
| `flowformer`, `gma`, `gmflow`, `memflow`, ... | ptlflow | other SOTA models |
| `ptlflow` | ptlflow | any ptlflow model via `--model <name>` |
| `dummy` | — | zero flow (tests) |

```bash
fav compute-flow --frames frames/ --out flow/ --backend sea_raft
fav train ... data.flow_backend=sea_raft
```

RAFT stays the default so things work without extra installs; `sea_raft` is the
quality pick once `ptlflow` is available. Output `.flo`/`.pgm` is identical
across backends, so datasets and the stylize pipeline are unaffected by the choice.

## Train a per-style model

```bash
fav train --config configs/train_mixed.yaml \
    style_image=styles/mystyle.jpg data.image_dir=/data/coco \
    loss.loss_network=models/vgg16_caffe.pt
```

Defaults equal the paper settings (Adam 1e-3, batch 4, content relu3_3, style
relu1_2/2_2/3_3/4_3, `tanh×150`, TV 1e-6). The mixed config uses the multi-frame
recurrent schedule `0:1,50000:2,60000:4`. Each run produces one `.pt` per style.

## Stylize a video

```bash
fav stylize --config configs/stylize.yaml \
    model_vid=checkpoint.pt input_pattern='frames/frame_%05d.png' \
    flow_pattern='flow/backward_[%d]_{%d}.flo' \
    occlusions_pattern='flow/reliable_[%d]_{%d}.pgm' output_prefix='out/stylized'
```

The first frame is stylized by `self` (the video model with an all-occluded
prior) or a separate image model; subsequent frames warp the previous output by
the backward flow, mask by the certainty, and re-stylize.

## 360 / spherical video

Cube-face geometry (`fav.vr.perspective`, `fav.vr.cubemap`) and a per-face
temporal stylizer (`fav.vr.stylize_vr`) are provided; equirect↔cubeface uses
`py360convert` (replacing Transform360). Cross-face seam gradient-blending is the
remaining VR refinement for this phase.

## Module map (port ↔ legacy)

| `fav` | legacy |
|---|---|
| `preprocess` | `fast_artistic_video/preprocess.lua` |
| `models/generator` | `models_video.lua` |
| `warp/grid_sample` | `stnbdhw/BilinearSamplerBDHW` |
| `warp/flow_io` | `flowFileLoader.lua` |
| `occlusion/consistency` | `consistencyChecker/*.cpp` |
| `losses/*` | `PerceptualCriterion/ContentLoss/StyleLoss/GramMatrix/TotalVariation.lua` |
| `flow/*` | DeepFlow / FlowNet2 scripts |
| `data/*` | `DataLoader_video_{fake,real}.lua`, `video_dataset/*` |
| `train/loop` | `train_video.lua` |
| `infer/*` | `fast_artistic_video{,_core}.lua` |
| `vr/*` | `fast_artistic_video_vr.lua`, `vr_helper.lua` |

## Tests

```bash
pytest -q
```

Faithfulness is checked via invariants and hand numerics (a live Torch7 golden is
not runnable in 2026): preprocess round-trip, warp == integer-shift reference,
occlusion == analytic/loop reference, Gram hand-computed, TV grad == legacy
scatter, warp↔synthetic-shift cross-check, and end-to-end smoke train/stylize.
