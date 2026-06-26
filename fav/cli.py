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

    def sample_images(batch):
        idxs = torch.randint(0, len(img_source), (batch,))
        return torch.stack([img_source[int(i)] for i in idxs]).to(device)

    def data_fn(it):
        source = "single_image" if it < cfg.data.single_image_until else mix.sample()
        num = value_at(frame_steps, it)
        if source == "single_image":
            num = 1
        imgs_raw = sample_images(cfg.batch_size)
        mode = source
        if source == "video":
            if not warned["video"]:
                print("NOTE: no real-video source configured; 'video' falls back to 'shift'.")
                warned["video"] = True
            mode = "shift"
        imgs, flows, certs = synth.sample(mode, imgs_raw, num)
        return source, imgs, flows, certs

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


def run_compute_flow(frames_dir, out_dir, backend="raft", pattern="frame_%05d.png", start=1):
    from fav.flow import build_estimator, compute_pair, write_pair
    from PIL import Image
    import numpy as np

    est = build_estimator(backend)

    def load(i):
        p = Path(frames_dir) / (pattern % i)
        if not p.exists():
            return None
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    i = start
    prev = load(i)
    written = 0
    while True:
        cur = load(i + 1)
        if cur is None:
            break
        flow_uv, reliable = compute_pair(est, prev, cur)
        write_pair(out_dir, i + 1, i, flow_uv, reliable)
        prev = cur
        i += 1
        written += 1
    return written


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
    p_flow.add_argument("--backend", default="raft")
    p_flow.add_argument("--pattern", default="frame_%05d.png")
    p_flow.add_argument("--start", type=int, default=1)

    p_conv = sub.add_parser("convert-vgg", help="convert vgg16.t7 -> .pt loss net")
    p_conv.add_argument("t7")
    p_conv.add_argument("out")

    for name, cfg_cls in (("train", C.TrainConfig), ("stylize", C.StylizeConfig),
                          ("stylize-vr", C.VRConfig)):
        sp = sub.add_parser(name, help=f"{name} (config-driven)")
        sp.add_argument("--config", default=None)
        sp.add_argument("--device", default=None)
        sp.add_argument("overrides", nargs="*", help="dotted key=value overrides")

    args = parser.parse_args(argv)

    if args.cmd == "compute-flow":
        n = run_compute_flow(args.frames, args.out, args.backend, args.pattern, args.start)
        print(f"wrote flow/occlusion for {n} frame pairs")
        return
    if args.cmd == "convert-vgg":
        from fav.conversion import convert_vgg16_t7

        out = convert_vgg16_t7(args.t7, args.out)
        print(f"wrote {out}")
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
        raise SystemExit("stylize-vr CLI wiring pending; use fav.vr.stylize_vr API for now")


if __name__ == "__main__":
    main()
