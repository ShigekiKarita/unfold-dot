from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='unfold_dot',
    ext_modules=[
        CUDAExtension(
            'unfold_dot_cuda', [
                'src/unfold_dot_cuda.cpp',
                'src/unfold_dot_cuda_kernel.cu',
            ],
            extra_compile_args={
                "cxx": ["-Wall", "-Wextra", "-O2"],
                "nvcc": ['-Xcompiler="-Wall"', '-Xcompiler="-Wextra"', '-Xcompiler="-O2"', "-O2"]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    test_suite="test.py"
)
