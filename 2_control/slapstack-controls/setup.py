from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

from platform import system
from os import remove
from os.path import sep
import numpy

if system() == "Windows":
    f = open('setup.cfg', 'w')
    f.write("[build]\ncompiler=msvc")
    f.close()

cython_root_dir = sep.join(["slapstack_controls", "extensions/"])

exts = [Extension(f"{cython_root_dir}*", [f'{cython_root_dir}*.pyx'],
                  include_dirs=[numpy.get_include()])]

setup(name='slapstack-controls',
      version='0.0.1',
      install_requires=['slapstack', 'numpy'],
      ext_modules=cythonize(exts, annotate=False),
      packages=find_packages(),
      )

if system() == "Windows":
    remove("setup.cfg")
