import numpy
import sysconfig
from pathlib import Path
from setuptools import setup, Extension, find_packages

# Header paths
python_include_path = sysconfig.get_path("include")
numpy_include_path = numpy.get_include()

module = Extension(
    name="min_norm_point.min_norm_point_module",
    sources=["min_norm_point/min_norm_point_module.c"],   # relative path
    include_dirs=[
        numpy_include_path,
        python_include_path,
        "/usr/include/glpk",
    ],
    libraries=["glpk"],
    library_dirs=[
        "/usr/lib/",
        "/usr/lib/x86_64-linux-gnu/",
        "/usr/local/lib/",
    ],
)

setup(
    name="min_norm_point",
    version="1.0.0",
    description="Python wrapper around a C extension for min norm point",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[module],
    include_package_data=True,
)
