# Minimum-Norm Point C Module

C module to compute the minimum-norm point $y$ in an $n$-dimensional Euclidean space $\mathbb{R}^n$ from a given point $x \in \mathbb{R}^n$ to a convex polyhedral set $P \subseteq \mathbb{R}^n$ defined as the intersection of $m$ closed halfspaces described by a system of linear inequalities $Ax + b \leq 0$.

## Shell Commands
```sh
sudo apt-get install glpk-utils libglpk-dev glpk-doc
sudo apt-get install build-essential
cd ./src/min_norm_point_C/
python setup.py build_ext --inplace
```

## Folder Structure
The folder `min_norm_point_C/` and its sub-folder `min_norm_point_C/min_norm_point/` must contain a total of **7 files**:

- **min_norm_point_module.c**: C file containing all needed C functions including the main `minimum_norm_points_to_polyhedra_py` (Python-compatible).
- **min_norm_point.py**: Python wrapper re-defining the function for use outside this folder.
- **\_\_init\_\_.py**: Initialization file to allow importing `minimum_norm_points_to_polyhedra`.
- **setup.py**: C-to-Python configuration for exposing the C function to Python.
- **pyproject.toml**: Required for C compilation, listing dependencies and backend.
- **min_norm_point_module.<cpython-…>.so** *(optional)*: Generated shared object. Recommended to remove and regenerate using:
  ```sh
  python setup.py clean --all
  rm -rf build
  python setup.py build_ext --inplace
  ```
- **README.md**: This file.

## Requirements
Please ensure you have:

- Installed **GLPK** for `min_norm_point_module.c`:
  ```sh
  sudo apt-get install glpk-utils libglpk-dev glpk-doc
  ```
- Correctly set up **setup.py** with paths to:
  - Python.h
  - numpy/arrayobject.h
  - glpk.h
- Generated the shared object file:
  ```sh
  python setup.py build_ext --inplace
  ```
  Ensure `build-essential` is installed:
  ```sh
  sudo apt-get install build-essential
  ```

You can verify that the `.so` file was generated correctly using:
```sh
ldd min_norm_point_module.<cpython-...>.so
```
Check that required modules (e.g., `libglpk.so`) appear.

## Usage
If everything is correctly configured, import the main function in Python:
```python
from min_norm_point import minimum_norm_points_to_polyhedra
```

Enjoy!
