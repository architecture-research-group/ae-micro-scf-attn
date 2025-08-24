from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="xor_popcount_cuda",
    ext_modules=[
        CUDAExtension(
            "xor_popcount_cuda",
            ["binding.cpp", "xor_popcount_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
