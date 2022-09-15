from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

from platform import system
from os.path import join, sep
from os import listdir, remove
import glob

if system() == "Windows":
    f = open('setup.cfg', 'w')
    f.write("[build]\ncompiler=msvc")
    f.close()

cython_root_dir = join("slapstack", "extensions")
use_case_target_dir = join("slapstack", "WEPAStacks")

use_case_files = [
    "Initial_fill_lvl.json",
    "Orders_v5.json",
    join("layouts", "layout1_middle_aisles.csv"),
    join("layouts", "layout1_mini_middle_aisles.csv")
]

with open("README.md", 'r') as f:
    long_description = f.read()

with open("MANIFEST.in", "w") as f:
    for file_name in use_case_files:
        dest = join(use_case_target_dir, file_name)
        f.write(f"include {dest}\n")
    for file_name in listdir(cython_root_dir):
        if file_name.endswith(".pyx"):
            dest = join(cython_root_dir, file_name)
            f.write(f"include {dest}\n")
    f.write(f'recursive-include slapstack *.pyd')

exts = []
for cython_source in glob.glob(join(cython_root_dir, "*.pyx")):
    exts.append(
        Extension(
            name=(cython_source.rsplit('.', 1)[0]
                  .replace('\\', '.').replace('/', '.')),
            sources=[cython_source]
        )
    )

setup(name='slapstack',
      version='0.0.24',
      python_requires='>3.6.8',
      install_requires=['gym', 'numpy', 'joblib', 'cython'],
      ext_modules=cythonize(
          exts,
          annotate=False,
          language="c++"
      ),
      description="An Event Discrete Simulation Framework for "
                  "Block-Stacking Warehouses.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={'slapstack': ['WEPAStacks/*.json',
                                  'WEPAStacks/layouts/*.csv',
                                  'extensions/*.pyd']},
      include_package_data=True
      )

if system() == "Windows":
    remove("setup.cfg")
