import os
from setuptools import setup, find_namespace_packages

requirements = open('requirements.txt').readlines()
with open(os.path.normpath(os.path.join(__file__, '../VERSION'))) as f:
    __version__ = f.readline().strip()


setup(name='pet_detection',
      version=__version__,
      description='A repo for Pet detection application',
      author='Omer Danziger',
      author_email='Omerdan03@gmail.com',
      packages=find_namespace_packages(),
      install_requires=requirements,
      entry_points={'console_scripts': []},
      )
