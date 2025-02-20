from setuptools import setup, Extension
import numpy
import sysconfig

# Chemins vers Python.h et numpy/arrayobject.h
python_include_path = sysconfig.get_path('include')
numpy_include_path = numpy.get_include()

module = Extension('min_norm_point_module', 
    sources=['min_norm_point_module.c'], 
    include_dirs=[python_include_path, numpy_include_path, '/usr/include/glpk'],  # Chemin des en-tetes necessaires: Python, Numpy et glpk.h
    libraries=['glpk'],  # Lien avec la bibliotheque GLPK
    library_dirs=['/usr/lib/', '/usr/lib/x86_64-linux-gnu/', '/usr/local/lib/']   # Chemin des fichiers libglpk.so
)

setup(
    name='min_norm_point_module',
    version='1.0',
    description='Un module qui rend minimum_norm_points_to_polyhedra compatible avec Python.',
    ext_modules=[module]
)
