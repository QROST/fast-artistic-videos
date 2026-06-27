"""Command-line interface: build-dataset, compute-flow, train, stylize, stylize-vr.

Thin argparse wrapper over reusable ``run_*`` helpers (which the tests call
directly). Configs come from YAML (``--config``) with ``key=value`` / dotted CLI
overrides; every default matches the verified legacy default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fav import config as C
from fav.device import select_device


# --- helpers shared by CLI and tests --------------------------------------- #

def build_loss_net_and_criterion(loss_cfg, model_cfg, style_image_rgb=None):
    """Build the VGG loss net + PerceptualCriterion; load weights if a .pt path."""
    from fav.losses.vgg_loss_net import build_vgg16_loss_net
    from fav.losses.perceptual import PerceptualCriterion
    from fav.preprocess import preprocess

    weights = loss_cfg.loss_network
    weights = weights if (weights and weights.endswith(".pt") and Path(weights).exists()) else None
    if weights is None:
        print("WARNING: no converted VGG .pt weights given; using random loss-net "
              "weights (pipeline is valid; real style fidelity needs converted weights).")
    net = build_vgg16_loss_net(loss_cfg.content_layers, loss_cfg.style_layers, weights)
    crit = PerceptualCriterion(
        net, loss_cfg.content_layers, loss_cfg.content_weights,
        loss_cfg.style_layers, loss_cfg.style_weights, agg_type=loss_cfg.style_target_type,
    )
    if style_image_rgb is not None:
        crit.set_style_target(preprocess(style_image_rgb, model_cfg.preprocessing))
    return net, crit


def build_data_fn(cfg, device="cpu"):
    """Build a ``data_fn(iteration)`` from the config's sources.

    Uses ``ImageFolderSource`` for the synthetic motion sources. If no real-video
    source is configured, ``video`` samples fall back to ``shift`` (logged once).
    """
    from fav.data.image_folder import ImageFolderSource
    from fav.data.synthetic import SyntheticSource
    from fav.data.mixed import DataMix
    from fav.train.schedules import parse_step_schedule, value_at

    mix = DataMix(cfg.data.data_mix)
    crop = int(cfg.data.train_img_size.split(":")[0])
    img_source = ImageFolderSource(cfg.data.image_dir, cfg.data.source_img_size, crop)
    synth = SyntheticSource()
    frame_steps = parse_step_schedule(cfg.data.num_frame_steps)
    warned = {"video": False}

    # Real-video source, if a clip directory is configured and the mix uses it.
    video_source = None
    if mix.needs_real_video and cfg.data.video_dir:
        from fav.data.video_clips import VideoClipsSource
        from fav.flow import build_estimator

        est = build_estimator(cfg.data.flow_backend, model=cfg.data.flow_model or None)
        video_source = VideoClipsSource(cfg.data.video_dir, est, crop=crop)

    gen = torch.Generator().manual_seed(cfg.seed) if cfg.seed else None

    def sample_images(batch):
        idxs = torch.randint(0, len(img_source), (batch,), generator=gen)
        return torch.stack([img_source[int(i)] for i in idxs]).to(device)

    def data_fn(it):
        source = "single_image" if it < cfg.data.single_image_until else mix.sample()
        num = value_at(frame_steps, it)
        if source == "single_image":
            num = 1
        if source == "video" and video_source is not None:
            imgs, flows, certs = video_source.sample(num, cfg.batch_size)
            imgs = [x.to(device) for x in imgs]
            flows = [f.to(device) for f in flows]
            certs = [c.to(device) for c in certs]
            return source, imgs, flows, certs
        mode = source
        if source == "video":
            if not warned["video"]:
                print("NOTE: no real-video source configured (data.video_dir empty); "
                      "'video' falls back to 'shift'.")
                warned["video"] = True
            mode = "shift"
        imgs_raw = sample_images(cfg.batch_size)
        imgs, flows, certs = synth.sample(mode, imgs_raw, num)
        # Return the actual mode so the logged/bucketed source label matches the
        # data produced (a 'video' fallback yields 'shift' data, not 'video').
        return mode, imgs, flows, certs

    return data_fn


def run_train(cfg, device=None, max_iters=None):
    from fav.models.generator import build_model
    from fav.train.loop import train
    from fav.train.checkpoint import save_checkpoint
    from PIL import Image
    import numpy as np

    device = select_device(device or cfg.device)
    model = build_model(cfg.model).to(device)
    style_rgb = None
    if cfg.style_image:
        with Image.open(cfg.style_image) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
        style_rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, crit = build_loss_net_and_criterion(cfg.loss, cfg.model, style_rgb)
    crit.loss_net.to(device)
    data_fn = build_data_fn(cfg, device)

    def on_step(it, m):
        if it % cfg.print_every == 0:
            print(f"iter {it}/{cfg.num_iterations} loss={m['loss']:.4f} "
                  f"(percep={m['percep']:.4f} pixel={m['pixel']:.4f}) src={m['source']}")
        if it % cfg.checkpoint_every == 0:
            save_checkpoint(f"{cfg.checkpoint_name}.pt", model, None, it, C.to_dict(cfg), [])

    history = train(model, crit, data_fn, cfg, device=str(device),
                    max_iters=max_iters or cfg.num_iterations, on_step=on_step)
    save_checkpoint(f"{cfg.checkpoint_name}.pt", model, None,
                    max_iters or cfg.num_iterations, C.to_dict(cfg), history)
    return history


def run_compute_flow(frames_dir, out_dir, backend="raft", pattern="frame_%05d.png", start=1,
                     estimator=None, model=None):
    from fav.flow import build_estimator, compute_pair, write_pair
    from fav.data.io_utils import load_rgb

    est = estimator if estimator is not None else build_estimator(backend, model=model)

    def load(i):
        p = Path(frames_dir) / (pattern % i)
        return load_rgb(p) if p.exists() else None

    i = start
    prev = load(i)
    written = 0
    while prev is not None:
        cur = load(i + 1)
        if cur is None:
            break
        flow_uv, reliable = compute_pair(est, prev, cur)
        write_pair(out_dir, i + 1, i, flow_uv, reliable)
        prev = cur
        i += 1
        written += 1
    return written


def run_build_dataset(video_dir, out_dir, backend="raft", pattern="frame_%05d.png", start=1,
                      model=None):
    """Precompute flow + occlusion assets for every clip subdir under video_dir.

    Mirrors video_dataset/make_*: each clip's assets are written under
    out_dir/<clip_name>/ with the legacy filenames, ready for the stylize
    pipeline or asset-based training.
    """
    from fav.flow import build_estimator

    root = Path(video_dir)
    clips = [p for p in sorted(root.iterdir()) if p.is_dir()] or [root]
    est = build_estimator(backend, model=model)  # reuse one estimator across clips
    total = 0
    for clip in clips:
        sub_out = Path(out_dir) / clip.name if clip is not root else Path(out_dir)
        total += run_compute_flow(str(clip), str(sub_out), backend, pattern, start, estimator=est)
    return total


def run_stylize_vr(cfg, device=None):
    from fav.train.checkpoint import load_checkpoint
    from fav.models.generator import Generator
    from fav.vr.stylize_vr import stylize_faces_over_time, faces_to_equirect
    from fav.vr.cubemap import PROC_ORDER
    from fav.infer.stylize_planar import resolve_pattern
    from fav.warp.flow_io import read_flo, read_pgm, uv_to_dydx
    from fav.data.io_utils import load_rgb, save_rgb

    device = select_device(device or cfg.device)
    ckpt = load_checkpoint(cfg.model_vid, map_location=str(device))
    model = Generator(ckpt["config"]["model"]["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    faces = sorted(PROC_ORDER)
    start = cfg.continue_with

    def frame_path(idx, face):
        return cfg.input_pattern % (idx, face)

    n = 0
    while n < cfg.num_frames and Path(frame_path(start + n, faces[0])).exists():
        n += 1
    if n == 0:
        raise FileNotFoundError(f"no VR frames match {cfg.input_pattern} from {start}")

    faces_seq, flows_seq, certs_seq = [], [], []
    for t in range(n):
        idx = start + t
        faces_seq.append({f: load_rgb(frame_path(idx, f)).to(device) for f in faces})
        if t > 0:
            fl, ce = {}, {}
            for f in faces:
                flo = read_flo(resolve_pattern(cfg.flow_pattern, idx, idx - 1) % f).to(device)
                fl[f] = uv_to_dydx(flo).unsqueeze(0)
                cert_px = read_pgm(resolve_pattern(cfg.occlusions_pattern, idx, idx - 1) % f).float()
                ce[f] = (cert_px / 255.0).view(1, 1, *cert_px.shape).to(device)
            flows_seq.append(fl)
            certs_seq.append(ce)

    out_seq = stylize_faces_over_time(
        model, faces_seq, flows_seq, certs_seq, model_img=cfg.model_img,
        occlusions_min_filter=cfg.occlusions_min_filter,
        median_filter_size=cfg.median_filter, fill_occlusions=cfg.fill_occlusions,
    )

    paths = []
    for t in range(n):
        idx = start + t
        for f in faces:
            p = Path(f"{cfg.output_prefix}-{idx:05d}-{f}.png")
            save_rgb(p, out_seq[t][f])
            paths.append(p)
        if cfg.out_equi:
            equi = faces_to_equirect(out_seq[t], cfg.out_equi_h, cfg.out_equi_w)
            save_rgb(f"{cfg.output_prefix}-{idx:05d}-equi.png", equi)
    return paths


def run_stylize(cfg, device=None):
    from fav.train.checkpoint import load_checkpoint
    from fav.models.generator import Generator
    from fav.infer.stylize_planar import stylize_video

    device = select_device(device or cfg.device)
    ckpt = load_checkpoint(cfg.model_vid, map_location=str(device))
    arch = ckpt["config"]["model"]["arch"]
    model = Generator(arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return stylize_video(cfg, model, model_img=cfg.model_img, device=str(device))


# --- argparse ---------------------------------------------------------------- #

def _apply_overrides(cfg, items):
    overrides: dict = {}
    for item in items:
        key, _, val = item.partition("=")
        node = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return C.merge_overrides(cfg, overrides)


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fav", description="Fast Artistic Videos (PyTorch/MPS)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_flow = sub.add_parser("compute-flow", help="compute .flo + reliable .pgm for a frame dir")
    p_flow.add_argument("--frames", required=True)
    p_flow.add_argument("--out", required=True)
    p_flow.add_argument("--backend", default="raft", help="raft|sea_raft|flowformer|ptlflow|...")
    p_flow.add_argument("--model", default=None, help="specific model name (raft/ptlflow)")
    p_flow.add_argument("--pattern", default="frame_%05d.png")
    p_flow.add_argument("--start", type=int, default=1)

    p_ds = sub.add_parser("build-dataset", help="precompute flow+occlusion assets for clip subdirs")
    p_ds.add_argument("--video", required=True, help="dir of clip subdirectories")
    p_ds.add_argument("--out", required=True)
    p_ds.add_argument("--backend", default="raft", help="raft|sea_raft|flowformer|ptlflow|...")
    p_ds.add_argument("--model", default=None, help="specific model name (raft/ptlflow)")
    p_ds.add_argument("--pattern", default="frame_%05d.png")
    p_ds.add_argument("--start", type=int, default=1)

    p_conv = sub.add_parser("convert-vgg", help="convert vgg16.t7 -> .pt loss net")
    p_conv.add_argument("t7")
    p_conv.add_argument("out")

    p_bench = sub.add_parser("bench", help="benchmark generator forward+backward throughput")
    p_bench.add_argument("--size", type=int, default=256)
    p_bench.add_argument("--iters", type=int, default=20)
    p_bench.add_argument("--device", default=None)
    p_bench.add_argument("overrides", nargs="*", help="dotted key=value (e.g. precision=bf16 compile_model=true)")

    for name, cfg_cls in (("train", C.TrainConfig), ("stylize", C.StylizeConfig),
                          ("stylize-vr", C.VRConfig)):
        sp = sub.add_parser(name, help=f"{name} (config-driven)")
        sp.add_argument("--config", default=None)
        sp.add_argument("--device", default=None)
        sp.add_argument("overrides", nargs="*", help="dotted key=value overrides")

    args = parser.parse_args(argv)

    if args.cmd == "compute-flow":
        n = run_compute_flow(args.frames, args.out, args.backend, args.pattern, args.start,
                             model=args.model)
        print(f"wrote flow/occlusion for {n} frame pairs")
        return
    if args.cmd == "build-dataset":
        n = run_build_dataset(args.video, args.out, args.backend, args.pattern, args.start,
                              model=args.model)
        print(f"wrote flow/occlusion assets for {n} frame pairs across clips")
        return
    if args.cmd == "convert-vgg":
        from fav.conversion import convert_vgg16_t7

        out = convert_vgg16_t7(args.t7, args.out)
        print(f"wrote {out}")
        return
    if args.cmd == "bench":
        from fav.bench import benchmark

        cfg = C.TrainConfig()
        if args.overrides:
            cfg = _apply_overrides(cfg, args.overrides)
        result = benchmark(cfg, device=args.device, size=args.size, iters=args.iters)
        print("  ".join(f"{k}={v}" for k, v in result.items()))
        return

    cfg_cls = {"train": C.TrainConfig, "stylize": C.StylizeConfig, "stylize-vr": C.VRConfig}[args.cmd]
    cfg = cfg_cls()
    if args.config:
        cfg = C.merge_overrides(cfg, C.load_yaml(args.config))
    if args.overrides:
        cfg = _apply_overrides(cfg, args.overrides)
    if getattr(args, "device", None):
        cfg.device = args.device

    if args.cmd == "train":
        run_train(cfg)
    elif args.cmd == "stylize":
        paths = run_stylize(cfg)
        print(f"wrote {len(paths)} stylized frames")
    elif args.cmd == "stylize-vr":
        paths = run_stylize_vr(cfg)
        print(f"wrote {len(paths)} stylized cube-face frames")


if __name__ == "__main__":
    main()
