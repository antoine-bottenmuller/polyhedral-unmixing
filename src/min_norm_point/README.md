C module to compute the minimum-norm point $y$ in a $n$-dimensional Euclidean space $\mathbb{R}^n$ from a given point $x \in \mathbb{R}^n$ to a convex polyhedra $P \subseteq \mathbb{R}^n$ defined as the intersection of $m$ halfspaces described by a system of linear inequalities $Ax + b \leq 0$.

This folder "min_norm_point/" must contain a total of 7 files:
* "min_norm_point_module.c": the C file containing all the needed C functions and the main one "minimum_norm_points_to_polyhedra_py", which is Python-compatible ;
* "min_norm_point.py": the Python file using and re-defining the "minimum_norm_points_to_polyhedra_py" function which can be called and used outside this folder as a Python function ;
* "__init__.py": the initialization file to use this foler "min_norm_point/" as a Python library and directly import the function "minimum_norm_points_to_polyhedra" from min_norm_point ;
* "setup.py": the C-to-Python configuration file, allowing one to use "minimum_norm_points_to_polyhedra_py" function from the C file in an independant Python file as a Python function ;
* "pyproject.toml": needed file for the C compilation indicating the needed Python libraries and the backend for setup.py ;
* if it has already been generated, the "min_norm_point_module.<cpython-...>.so" for a Python use of the implemented functions in C: it is recomended to remove it (with: python setup.py clean --all ; rm -rf build) and generate it again with your own configurations in "setup.py" (with: python setup.py build_ext --inplace) ;
* this "README.md" file.

Please make sure you have:
* installed GLPK for the C module "min_norm_point_module.c": if it is not installed yet, you can use the command: sudo apt-get install glpk-utils libglpk-dev glpk-doc ;
* correctly set up the C-to-Python file "setup.py" with your own paths to the needed modules on your computer (Python.h, numpy/arrayobject.h, glpk.h) ;
* generated the "min_norm_point_module.<cpython-...>.so" file with the command: python setup.py build_ext --inplace (make sure you have build-essential installed: sudo apt-get install build-essential).

You can make sure that the .so file has been correctly generated and contains the needed modules by using the command: ldd min_norm_point_module.cpython-310-x86_64-linux-gnu.so, and checking if the asked modules appear (libglpk.so).

If everything is well configurated for your own use, you will only have to import the main function "minimum_norm_points_to_polyhedra" from this folder "min_norm_point" in your Python file or notebook, with simply: 
from min_norm_point import minimum_norm_points_to_polyhedra

For any question, you can contact the author at: antoine.bottenmuller@gmail.com

Enjoy!
