from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


WITHOUT_CUDA = False
if not torch.cuda.is_available():
    WITHOUT_CUDA = True

SOURCES = [
    'csrc/croco.cpp',
    'csrc/cpu/croco_function_cpu.cpp',
]
if not WITHOUT_CUDA:
    SOURCES.append('csrc/cuda/croco_function_cuda.cu')
EXTENSION = CUDAExtension if not WITHOUT_CUDA else CppExtension
DEFINE_MACROS = [] if not WITHOUT_CUDA else [("WITHOUT_CUDA", None)]
EXTRA_COMPILE_ARGS = {'cxx': ['-O3', '-g', '-fopenmp'], 'nvcc': ['-O3']} if not WITHOUT_CUDA else {'cxx': ['-O3', '-g']}

setup(
    name='croco',
    version='1.0.0',
    description='Cross-view coorperative reasoning backend with C implementation for pytorch',
    author='Junchen Zhu',
    author_email='junchen.zhu@hotmail.com',
    packages=find_packages(),
    ext_modules=[
        EXTENSION(
            name='croco._C', 
            sources=SOURCES,
            extra_compile_args=EXTRA_COMPILE_ARGS,
            define_macros=DEFINE_MACROS,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)