from setuptools import Extension, setup, find_packages
from pathlib import Path
import numpy

ROOT = Path(__file__).parent.resolve()


gradengc_module = Extension(
            name="gradengc",
            sources=[
                "Modules/gradientenginecustom.c",
                "Modules/foosum.c",
                "Modules/doublekeytreeops.c",
                "Modules/loadPolygonList.c",
                "Modules/point_orderer.c"
            ],
            include_dirs=[numpy.get_include(), f"{ROOT}/Modules"],
            depends=[
                "Modules/foosum.h",
                "Modules/doublekeytreedata.h",
                "Modules/doublekeytreeops.h",
                "Modules/loadPolygonList.h",
                "Modules/point_orderer.h"
            ]
        )

setup(
    name='GradientEngine',
    version='0.0.1',
    url='',
    license='',
    author_email='',
    packages=find_packages(),
    ext_modules=[gradengc_module]
)
