from setuptools import Extension
from setuptools import setup

import numpy as np
import os
from Cython.Build import cythonize

# skgrf project directory
top = os.path.dirname(os.path.abspath(__file__))

# include skgrf, grf, and numpy headers
# requires running buildpre.py to find src in this location
include_dirs = [
    top,
    os.path.join(top, "skgrf"),
    *[d[0] for d in os.walk(os.path.join(top, "skgrf", "grf", "src"))],
    *[d[0] for d in os.walk(os.path.join(top, "skgrf", "grf", "third_party"))],
    np.get_include(),
]


def find_pyx_files(directory, files=None):
    """Recursively find all Cython extension files.

    :param str directory: The directory in which to recursively crawl for .pyx files.
    :param list files: A list of files in which to append discovered .pyx files.
    """
    if files is None:
        files = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            find_pyx_files(path, files)
    return files


def create_extension(module_name, additional_sources=None):
    """Create a setuptools build extension for a Cython extension file.

    :param str module_name: The name of the module
    :param list(str) additional_sources: A list of additional source filenames
    """
    path = module_name.replace(".", os.path.sep) + ".pyx"
    additional_sources = additional_sources or []
    return Extension(
        module_name,
        sources=[path] + additional_sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++11", "-Wall"],
        extra_link_args=["-std=c++11", "-g"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )


# Workaround for https://github.com/grf-labs/grf/issues/1004
# TODO consider moving cpp files out pf pxd and here instead
additional_sources = [
    os.path.join(
        top, "skgrf", "grf", "src", "prediction", "CausalSurvivalPredictionStrategy.cpp"
    )
]
ext_modules = [create_extension("skgrf.grf", additional_sources)]

setup(
    ext_modules=cythonize(
        ext_modules,
        gdb_debug=False,
        force=True,
        annotate=False,
        compiler_directives={"language_level": "3"},
    )
)


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": ext_modules})
