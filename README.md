# polyhedral-unmixing

Code for blind polyhedral unmixing of spectral images.

Example with the application of the method on the Samson dataset (notebook "tests_samson.ipynb").

The repository is organised as follows:
* samson: folder containing the Samson dataset (Y) and the ground truth (endmembers M and abundances A) in Matlab format ;
* src: folder containing the main code to use outside (proba_map.py for the main unmixing functions, functions.py for the minimum-norm point algorithm in Python, and min_norm_point_C for the minimum-norm point algorithm implemented in C) ;
* tests_samson.ipynb: notebook for the use and the application of the functions in src to the Samson dataset, with main results displayed ;
* requirements.txt: list of required Python packages ;
* README.md: this current file containing informations about this repository.

The method and main results are presented in:

Bottenmuller, A.; Magaud, F.; Demortière, A.; Decencière, E. and Dokladal, P. (2025). Euclidean Distance to Convex Polyhedra and Application to Class Representation in Spectral Images. In Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods, ISBN 978-989-758-730-6, ISSN 2184-4313, pages 192-203. 

