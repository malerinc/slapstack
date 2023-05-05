from setuptools import setup, find_packages

from os.path import join

root = ''  # dirname(abspath(__file__))

use_case_target_wepastacks = join(
    root, "slapstack", "use_cases", "wepastacks")
use_case_target_crossstacks = join(
    root, "slapstack", "use_cases", "crossstacks")

use_case_files = [
    "1_layout.csv",
    "2_orders.json",
    "3_initial_fill_lvl.json"
]

with open("README.md", 'r') as f:
    long_description = f.read()


with open("MANIFEST.in", "w") as f:
    f.write("include")
    for uc_dir in [use_case_target_wepastacks, use_case_target_crossstacks]:
        for file_name in use_case_files:
            dest = join(uc_dir, file_name)
            f.write(f" {dest}")

setup(name='slapstack',
      version='0.1.1',
      python_requires='>3.6.8',
      install_requires=[
          'gym', 'numpy', 'joblib', 'pandas', 'scipy'
      ],
      description="An Event Discrete Simulation Framework for "
                  "Block-Stacking Warehouses.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={
          'slapstack': [
              'use_cases/wepastacks/*.json',
              'use_cases/wepastacks/*.csv',
              'use_cases/crossstacks/*.json',
              'use_cases/crossstacks/*.csv'
          ]
      },
      include_package_data=True
)
