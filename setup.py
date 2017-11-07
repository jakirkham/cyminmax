#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import setuptools
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand

from distutils.sysconfig import get_config_var, get_python_inc

import versioneer


class PyTest(TestCommand):
    description = "Run test suite with pytest"

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        sys.exit(pytest.main(self.test_args))


with open("README.rst") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "cython>=0.25.2",
]

install_requirements = [
    # TODO: put package install requirements here
]

test_requirements = [
    "pytest",
]

cmdclasses = {
    "test": PyTest,
}
cmdclasses.update(versioneer.get_cmdclass())


if not (({"develop", "test"} & set(sys.argv)) or
    any([v.startswith("build") for v in sys.argv]) or
    any([v.startswith("bdist") for v in sys.argv]) or
    any([v.startswith("install") for v in sys.argv])):
    setup_requirements = []


include_dirs = [
    os.path.dirname(get_python_inc()),
    get_python_inc()
]
library_dirs = list(filter(
    lambda v: v is not None,
    [get_config_var("LIBDIR")]
))

headers = []
sources = glob("src/*.pxd") + glob("src/*.pyx")
libraries = []
define_macros = []
extra_compile_args = []
cython_directives = {}
cython_line_directives = {}


ext_modules = [
    Extension(
        "cyminmax",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language="c"
    )
]
for em in ext_modules:
    em.cython_directives = dict(cython_directives)
    em.cython_line_directives = dict(cython_line_directives)


setup(
    name="cyminmax",
    version=versioneer.get_version(),
    description="A minmax implementation in Cython for use with NumPy",
    long_description=readme,
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    url="https://github.com/jakirkham/cyminmax",
    cmdclass=cmdclasses,
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    setup_requires=setup_requirements,
    install_requires=install_requirements,
    headers=headers,
    ext_modules=ext_modules,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="cyminmax",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    tests_require=test_requirements
)
