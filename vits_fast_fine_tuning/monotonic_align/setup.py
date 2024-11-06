from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

setup(
  name = 'monotonic_align',
  # ext_modules = cythonize("core.pyx"),
  ext_modules = cythonize(os.path.join(os.path.dirname(__file__), "core.pyx")),
  include_dirs=[numpy.get_include()],
  options={
        'build_ext': {
            'build_lib': 'VITS-fast-fine-tuning/monotonic_align'
        }
    }
)
