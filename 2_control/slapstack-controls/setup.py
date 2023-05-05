from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='slapstack-controls',
      version='0.1.1',
      python_requires='>3.6.8',
      install_requires=[
            'slapstack', 'numpy'
      ],
      description="Storage Strategies for the SLAPStack simulation Framework.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      )
