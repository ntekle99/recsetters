"""Build the fused CUDA kernels.

    pip install -e cuda_ext/          # from the repo root, on a CUDA machine

Requires a CUDA toolkit whose nvcc matches the CUDA version PyTorch was built
against (``python -c "import torch; print(torch.version.cuda)"``). Without a
toolkit the package still installs its Python half and ops.py falls back to
PyTorch implementations.
"""

import os

from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join("csrc", "bindings.cpp"),
    os.path.join("csrc", "dot_compress_cuda.cu"),
    os.path.join("csrc", "quant_embedding_cuda.cu"),
]

ext_modules = []
cmdclass = {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        ext_modules = [
            CUDAExtension(
                name="recsetters_cuda",
                sources=SOURCES,
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math", "--extended-lambda"],
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
    else:
        print("[recsetters] no CUDA device visible; installing Python fallbacks only. "
              "Set FORCE_CUDA=1 to build anyway (e.g. on a build node).")
except ImportError:
    print("[recsetters] torch not importable at build time; skipping the CUDA extension.")

setup(
    name="recsetters-cuda",
    version="0.1.0",
    description="Fused CUDA kernels, INT8 quantisation and cascade ranking for RecSetters",
    packages=["cuda_ext"],
    package_dir={"cuda_ext": "."},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
)
