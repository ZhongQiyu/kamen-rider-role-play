
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_add',
    ext_modules=[
        CUDAExtension('cuda_add', [
            'cuda_add.cpp',
            'cuda_add_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
