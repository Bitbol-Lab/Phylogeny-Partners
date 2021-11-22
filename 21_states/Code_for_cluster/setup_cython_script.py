from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "analyse_sequence",
        ["cython/analyse_sequence.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
   # Extension("MI_Method_cython_2_state",["MI_Method_cython_2_state.pyx"])  
]

setup(
    ext_modules=cythonize(ext_modules),
)


