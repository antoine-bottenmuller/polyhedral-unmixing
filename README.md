# polyhedral-unmixing

Code for blind polyhedral unmixing of spectral images.

Example with the application of the method on the Samson and Battery datasets (notebooks "example_<...>.ipynb").

The repository is organised as follows:
* datasets: folder containing the Samson dataset (Y) with the ground truth (endmembers M and abundances A) in Matlab format, and an original grayscale image of the chemical medium of a Lithium-ion battery captured by X-ray nano-CT in Numpy format ;
* src: folder containing the main code to use outside (unmixing.py for the main unmixing functions, min_norm_point_PYTHON.py for the minimum-norm point algorithm in Python, and the min_norm_point folder for the minimum-norm point algorithm implemented in C - must be compiled!) ;
* example_<...>.ipynb: two notebooks (one detailed and well commented version, and one short version) for the use and the application of the functions in src to the Samson and Battery datasets, with main results displayed ;
* requirements.txt: list of the required Python packages ;
* README.md: this current file containing informations about this repository.

To install the required Python libraries:
* pip install -r requirements.txt
* pip install ./src/min_norm_point

The method and main results are presented in:

Bottenmuller, A.; Magaud, F.; Demortière, A.; Decencière, E. and Dokladal, P. (2025). Euclidean Distance to Convex Polyhedra and Application to Class Representation in Spectral Images. In Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods, ISBN 978-989-758-730-6, ISSN 2184-4313, pages 192-203 (https://hal.science/hal-05001289v1).

