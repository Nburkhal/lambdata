"""
A collection of Data Science helper functions
@author: Nick Burkhalter
@url: https://github.com/Nburkhal/lambdata
"""

# Always prefer setuptools over distutils
import setuptools

REQUIRED = [
    'numpy',
    'pandas'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lambdata_nburkhal',
    version = '0.0.2',
    description='A collection of Data Science helper functions',
    long_description= long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=REQUIRED,
    author = 'Nick Burkhalter',
    author_email = 'nburkhal.nb@gmail.com',
    license='MIT'
)