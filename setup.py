import os
import shutil
from setuptools import Extension, setup
from Cython.Build import cythonize

# Clean previously generated files
def clean():
    for file_name in ['test.cpp', 'wrapper.cpp', 'wrapper.pyd', 'test.pyd']:
        if os.path.exists(file_name):
            os.remove(file_name)
    if os.path.exists('build'):
        shutil.rmtree('build')

clean()
import numpy as np


ext_modules = [
    Extension(
        "Runner_Edge",                           # Name of the module
        sources=["Runner_Edge.cpp", "Runner_Edge.pyx"],  # Your C++ and Cython sources
        include_dirs=[np.get_include(), "."],  # Include NumPy's header directory and current directory
        language="c++"                    # Specifies that we are using C++
    )
]

setup(
    name="Runner_Edge",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
)
