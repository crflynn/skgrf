import shutil
from pathlib import Path

top = Path(__file__).parent


def copy_grf_source():
    """Copy the grf cpp source, following symlinks."""
    grf = top / "grf"
    src = grf / "core"
    dst = top / "skgrf" / "grf"
    try:
        shutil.rmtree(dst)
    except FileNotFoundError:
        pass
    shutil.copytree(src, dst, symlinks=False)

    additional_files = ["COPYING", "README.md", "REFERENCE.md"]
    for f in additional_files:
        shutil.copyfile((grf / f).absolute(), (dst / f).absolute())


copy_grf_source()
