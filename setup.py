import os
from setuptools import setup, find_namespace_packages

requirements = open('requirements.txt').readlines()
with open(os.path.normpath(os.path.join(__file__, '../VERSION'))) as f:
    __version__ = f.readline(0)


setup(name='robotic_car_GUI',
      version=__version__,
      description='A repo for Pet detection program',
      author='Omer Danziger',
      author_email='Omerdan03@gmail.com',
      packages=find_namespace_packages(),
      install_requires=requirements,
      entry_points={'console_scripts': []},
      )
