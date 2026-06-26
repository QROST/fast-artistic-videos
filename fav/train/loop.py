"""Training step and loop, a faithful port of ``train_video.lua``.

The recurrent rollout is the load-bearing detail: intermediate frames are
generated as forward-only context (under ``no_grad``, reproducing the legacy
``out2:clone()`` graph break), and the loss + single ``backward()`` are computed
only on the final frame. The total loss is

    percep_loss_weight * perceptual(out_last, content_last)
  + pixel_loss_weight  * MSE(out_last * cert, warped_prev * cert)
  + tv_strength        * TV(out_last)

Data is supplied by a ``data_fn(iteration) -> (source, imgs_raw, flows, certs)``
callback (imgs_raw are RGB [0,1]); the loop preprocesses to VGG space, so the
loop itself is decoupled from how tuples are produced.
"""

from __future__ import annotations

import torch

from fav.losses.temporal import temporal_pixel_loss, tv_penalty
from fav.occlusion.filters import min_filter
from fav.preprocess import get_methods
from fav.warp.grid_sample import warp


def _generate_fill(cert, fill_occlusions, preprocess_fn):
    """Antimask fill for occluded regions (vgg-mean=0 or uniform random)."""
    if fill_occlusions == "vgg-mean":
        return torch.zeros_like(cert.expand(cert.shape[0], 3, *cert.shape[2:]))
    if fill_occlusions == "uniform-random":
        b, _, h, w = cert.shape
        rnd = preprocess_fn(torch.rand(b, 3, h, w, device=cert.device, dtype=cert.dtype))
        return rnd * (1.0 - cert)
    raise ValueError(f"unknown fill_occlusions {fill_occlusions!r}")


def _first_frame(model, image_model, source, frame0_pre):
    """Stylize the first frame: zeros (single_image) / self / separate model."""
    b, c, h, w = frame0_pre.shape
    if source == "single_image":
        return torch.zeros(b, 3, h, w, device=frame0_pre.device, dtype=frame0_pre.dtype)
    with torch.no_grad():
        if image_model is None:  # 'self': video net with an all-occluded prior
            prior = torch.zeros(b, 4, h, w, device=frame0_pre.device, dtype=frame0_pre.dtype)
            return model(torch.cat([frame0_pre, prior], dim=1))
        return image_model(frame0_pre)


def train_step(
    model,
    perceptual,
    optimizer,
    batch,
    cfg,
    image_model=None,
    device="cpu",
):
    """One optimization step. ``batch`` = ``(source, imgs_raw, flows, certs)``."""
    source, imgs_raw, flows, certs = batch
    preprocess_fn, _ = get_methods(cfg.model.preprocessing)

    imgs = [preprocess_fn(x.to(device)) for x in imgs_raw]
    flows = [f.to(device) for f in flows]
    certs = [min_filter(c.to(device), cfg.data.reliable_map_min_filter) for c in certs]

    num_steps = len(flows)
    out1 = _first_frame(model, image_model, source, imgs[0])

    optimizer.zero_grad(set_to_none=True)

    out2 = None
    warped_masked_last = None
    cert_last = certs[num_steps - 1]
    for i in range(num_steps):
        if out2 is not None:
            out1 = out2.detach()
        warped = warp(out1, flows[i])
        warped_masked = warped * certs[i]
        fill = _generate_fill(certs[i], cfg.data.fill_occlusions, preprocess_fn)
        inp = torch.cat([imgs[i + 1], warped_masked + fill, certs[i]], dim=1)
        is_last = i == num_steps - 1
        if is_last:
            out2 = model(inp)
            warped_masked_last = warped_masked
        else:
            with torch.no_grad():
                out2 = model(inp)

    content_target = imgs[num_steps]
    out2_masked = out2 * cert_last

    percep = cfg.loss.percep_loss_weight * perceptual(out2, content_target)
    pixel = cfg.loss.pixel_loss_weight * temporal_pixel_loss(
        out2_masked, warped_masked_last, cfg.loss.pixel_loss_type
    )
    tv = tv_penalty(out2, cfg.model.tv_strength)
    loss = percep + pixel + tv

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "percep": float(percep.detach()),
        "pixel": float(pixel.detach()),
        "tv": float(tv.detach()),
        "content": perceptual.total_content_loss,
        "style": perceptual.total_style_loss,
        "source": source,
    }


def train(model, perceptual, data_fn, cfg, device="cpu", image_model=None, max_iters=None,
          on_step=None):
    """Run the training loop, returning the per-step loss history.

    ``data_fn(iteration)`` yields a batch tuple. ``on_step(it, metrics)`` is an
    optional callback (e.g. for checkpointing/logging). This is the inner engine;
    the CLI wires ``data_fn`` to the real mix sampler + datasets.
    """
    from fav.train.schedules import parse_lr_schedule, value_at

    model.train()
    lr_sched = parse_lr_schedule(cfg.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_sched[0][1])

    n = max_iters if max_iters is not None else cfg.num_iterations
    history = []
    for it in range(1, n + 1):
        for g in optimizer.param_groups:
            g["lr"] = value_at(lr_sched, it)
        batch = data_fn(it)
        metrics = train_step(model, perceptual, optimizer, batch, cfg, image_model, device)
        metrics["iter"] = it
        history.append(metrics)
        if on_step is not None:
            on_step(it, metrics)
    return history
