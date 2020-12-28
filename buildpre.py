import os
import shutil

top = os.path.dirname(os.path.abspath(__file__))


def copy_grf_source():
    """Copy the ranger cpp source, following symlinks."""
    src = os.path.join(top, "grf", "core")
    dst = os.path.join(top, "skgrf", "ensemble", "grf")
    try:
        shutil.rmtree(dst)
    except FileNotFoundError:
        pass
    shutil.copytree(src, dst, symlinks=False)


copy_grf_source()
