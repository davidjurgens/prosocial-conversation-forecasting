from setuptools import find_packages, setup
import shutil
from pathlib import Path
import urllib.request
import os

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Code and models for the paper "Conversations Gone Alright: Quantifying and Predicting Prosocial Outcomes in Online Conversations" at the Web Conference 2021',
    author='Jiajun Bao*, Junjie Wu*, Yiming Zhang*, Eshwar Chandrasekharan, and David Jurgens.',
    license='',
)

urllib.request.urlretrieve("http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip", "models/mallet.zip")
shutil.unpack_archive(Path("models/mallet.zip"),
                      Path("models/mallet-2.0.8"))
os.remove(Path("models/mallet.zip"))
