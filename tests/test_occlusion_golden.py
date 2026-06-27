"""Golden-reference parity: Python occlusion port vs the legacy C++ consistencyChecker.

The C++ checker is standalone (no Torch), so we can compile it and assert the
Python port reproduces it bit-for-bit. This is the strongest faithfulness guard
for the occlusion module. Skips cleanly if a C++ toolchain isn't available.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from fav.warp.flow_io import write_flo, read_pgm
from fav.occlusion.consistency import check_consistency, compute_reliability

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "consistencyChecker"


@pytest.fixture(scope="module")
def checker_bin():
    if shutil.which("g++") is None or not (SRC / "consistencyChecker.cpp").exists():
        pytest.skip("no C++ toolchain / consistencyChecker sources")
    out = SRC / "consistencyChecker"
    if not out.exists():
        r = subprocess.run(
            ["g++", "-O3", "-fPIC", "consistencyChecker.cpp", "NMath.cpp", "-I.", "-o", "consistencyChecker"],
            cwd=SRC, capture_output=True, text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"consistencyChecker build failed: {r.stderr[-400:]}")
    return out


def _write_ppm(path, img):
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    h, w, _ = arr.shape
    Path(path).write_bytes(f"P6\n{w} {h}\n255\n".encode() + arr.tobytes())


def _run_cpp(checker_bin, d, flow1, flow2, content=None):
    write_flo(f"{d}/f1.flo", flow1)
    write_flo(f"{d}/f2.flo", flow2)
    args = [str(checker_bin), f"{d}/f1.flo", f"{d}/f2.flo", f"{d}/r.pgm"]
    if content is not None:
        _write_ppm(f"{d}/c.ppm", content)
        args.append(f"{d}/c.ppm")
    subprocess.run(args, check=True, capture_output=True)
    return read_pgm(f"{d}/r.pgm").float()


def test_golden_no_structure_bit_exact(checker_bin, tmp_path):
    for seed in range(10):
        torch.manual_seed(seed)
        h, w = 36, 44
        f1 = (torch.rand(2, h, w) - 0.5) * 8
        f2 = (torch.rand(2, h, w) - 0.5) * 8
        ref = _run_cpp(checker_bin, tmp_path, f1, f2)
        mine = check_consistency(f1, f2, structure=None)
        assert torch.equal(ref, mine), f"seed {seed}: no-structure mismatch"


def test_golden_with_structure_recursive_near_exact(checker_bin, tmp_path):
    # The recursive Deriche smoother reproduces the C++ structure term to within
    # float-rounding of the long recurrence (g++ -O3 FMA vs torch): bit-exact in
    # most cases, off by at most a few near-threshold pixels that min_filter
    # erodes. Require >=99.5% pixel agreement.
    for seed in range(10):
        torch.manual_seed(seed)
        h, w = 40, 40
        f1 = (torch.rand(2, h, w) - 0.5) * 8
        f2 = (torch.rand(2, h, w) - 0.5) * 8
        content = torch.rand(3, h, w)
        ref = _run_cpp(checker_bin, tmp_path, f1, f2, content)
        mine = compute_reliability(f1, f2, content_image=content, smooth="recursive")
        agree = (ref == mine).float().mean().item()
        assert agree >= 0.995, f"seed {seed}: structure agreement {agree:.4f} < 0.995"


def test_recursive_vs_gaussian_close():
    # The fast Gaussian option stays within a hair of the recursive default.
    torch.manual_seed(0)
    f1 = (torch.rand(2, 40, 40) - 0.5) * 8
    f2 = (torch.rand(2, 40, 40) - 0.5) * 8
    content = torch.rand(3, 40, 40)
    rec = compute_reliability(f1, f2, content_image=content, smooth="recursive")
    gauss = compute_reliability(f1, f2, content_image=content, smooth="gaussian")
    assert (rec == gauss).float().mean() > 0.99
