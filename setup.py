import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME


ROOT = Path(__file__).parent


def nccl_paths() -> tuple[list[str], list[str]]:
    """Return optional NCCL include and library directories from the environment."""
    prefixes = [
        os.environ.get("NCCL_HOME"),
        os.environ.get("NCCL_ROOT"),
        os.environ.get("NCCL_DIR"),
    ]
    if CUDA_HOME:
        prefixes.append(CUDA_HOME)

    include_dirs: list[str] = []
    library_dirs: list[str] = []
    for prefix in filter(None, prefixes):
        include = Path(prefix) / "include"
        lib64 = Path(prefix) / "lib64"
        lib = Path(prefix) / "lib"
        if include.is_dir():
            include_dirs.append(str(include))
        if lib64.is_dir():
            library_dirs.append(str(lib64))
        if lib.is_dir():
            library_dirs.append(str(lib))
    return include_dirs, library_dirs


include_dirs, library_dirs = nccl_paths()

setup(
    name="nccl-symm-mem",
    version="0.1.0",
    description="Out-of-tree NCCL symmetric-window allocation for PyTorch 2.9",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="nccl_symm_mem._C",
            # The implementation only calls host NCCL/CUDA APIs; it contains no
            # CUDA kernels. Building it as C++ avoids requiring the locally
            # installed nvcc to match the CUDA minor version used by the wheel.
            sources=[str(ROOT / "csrc" / "nccl_symm_mem.cpp")],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["nccl"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.9",
    install_requires=["torch>=2.9,<2.10"],
)
