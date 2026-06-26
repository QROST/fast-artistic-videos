"""Parse the legacy ``iter:value`` schedule strings.

Used for two things in ``train_video.lua``:

* ``-num_frame_steps`` e.g. ``"0:1,50000:2,60000:4"`` — the recurrent rollout
  length as a step function of the iteration.
* ``-learning_rate`` e.g. ``"1e-3"`` or ``"1e-3,50000:1e-4"`` — the base rate
  plus optional ``iter:rate`` breakpoints.

``value_at`` reproduces the legacy lookup: the value is that of the last
breakpoint whose iteration is strictly less than the current iteration
(``iteration > entry.iter``), matching the Lua loops.
"""

from __future__ import annotations


def parse_step_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse ``"0:1,50000:2"`` -> ``[(0, 1), (50000, 2)]`` (int values)."""
    out = []
    for part in spec.split(","):
        it, num = part.split(":")
        out.append((int(it), int(num)))
    return out


def parse_lr_schedule(spec: str) -> list[tuple[int, float]]:
    """Parse a learning-rate schedule.

    The first element may be a bare rate (no ``iter:``), treated as iter 0.
    e.g. ``"1e-3,50000:1e-4"`` -> ``[(0, 1e-3), (50000, 1e-4)]``.
    """
    parts = str(spec).split(",")
    out: list[tuple[int, float]] = []
    first = parts[0]
    if ":" in first:
        it, rate = first.split(":")
        out.append((int(it), float(rate)))
    else:
        out.append((0, float(first)))
    for part in parts[1:]:
        it, rate = part.split(":")
        out.append((int(it), float(rate)))
    return out


def value_at(schedule: list[tuple[int, float]], iteration: int):
    """Return the scheduled value at ``iteration`` (last breakpoint with it < iteration).

    Always returns at least the first entry's value (the legacy code seeds the
    value before the loop).
    """
    value = schedule[0][1]
    for it, val in schedule:
        if iteration > it:
            value = val
        else:
            break
    return value
