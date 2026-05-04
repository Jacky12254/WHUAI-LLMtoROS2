from setuptools import find_packages
from setuptools import setup

setup(
    name='fast_lio_sam_g',
    version='0.0.0',
    packages=find_packages(
        include=('fast_lio_sam_g', 'fast_lio_sam_g.*')),
)
