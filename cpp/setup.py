from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_LIB_NAME = 'transformer_fuse'

setup(
    name=f'{_LIB_NAME}_cpp',
    ext_modules=[
        CppExtension(f'{_LIB_NAME}_cpp', [f'{_LIB_NAME}.cpp']),
    ],
    cmdclass={'build_ext': BuildExtension},
)
