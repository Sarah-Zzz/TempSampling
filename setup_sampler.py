from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ColorSamplerCore',
    ext_modules=[
        CppExtension(
            name='ColorSamplerCore',
            sources=[
                'color_sampler_core.cpp',
            ],
            extra_compile_args = ['-fopenmp'],
            extra_link_args = ['-fopenmp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })