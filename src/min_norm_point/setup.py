from setuptools import setup, Extension
import numpy
import sysconfig

# Chemins d'en-tête vers Python.h et numpy/arrayobject.h
python_include_path = sysconfig.get_path('include')
numpy_include_path = numpy.get_include()

# Extension C
module = Extension(
    'min_norm_point_module', 
    sources=['min_norm_point_module.c'], 
    include_dirs=[python_include_path, numpy_include_path, '/usr/include/glpk'], 
    libraries=['glpk'], 
    library_dirs=['/usr/lib/', '/usr/lib/x86_64-linux-gnu/', '/usr/local/lib/']
)

# Configuration du package
setup(
    name='min_norm_point',
    version='1.0',
    description='Module Python avec une extension C pour le minimum norm point.',
    ext_modules=[module], 
    packages=['min_norm_point'], 
    include_package_data=True
)