import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME


ROOT = Path(__file__).parent
README = ROOT / "README.md"


def _nccl_header_version(prefix: str) -> int:
    """Return NCCL_VERSION_CODE from ``prefix/include/nccl.h``, or zero if absent."""
    header = Path(prefix) / "include" / "nccl.h"
    if not header.is_file():
        return 0
    match = re.search(
        r"^\s*#\s*define\s+NCCL_VERSION_CODE\s+(\d+)\s*$",
        header.read_text(encoding="utf-8", errors="ignore"),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else 0


def nccl_paths() -> tuple[list[str], list[str]]:
    """Return NCCL paths, preferring explicit prefixes then newest discovered headers."""
    explicit_prefixes = [
        os.environ.get("NCCL_HOME"),
        os.environ.get("NCCL_ROOT"),
        os.environ.get("NCCL_DIR"),
    ]
    discovered_prefixes = [
        str(Path(path) / "nvidia" / "nccl") for path in sys.path
    ]
    if CUDA_HOME:
        discovered_prefixes.append(CUDA_HOME)

    # A user-level wheel can be older than the system wheel. Sorting only
    # automatic candidates prevents it from silently shadowing newer headers;
    # NCCL_HOME/NCCL_ROOT/NCCL_DIR always retain explicit priority.
    prefixes = [prefix for prefix in explicit_prefixes if prefix]
    prefixes.extend(
        sorted(
            (prefix for prefix in discovered_prefixes if prefix),
            key=_nccl_header_version,
            reverse=True,
        )
    )

    include_dirs: list[str] = []
    library_dirs: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        normalized = str(Path(prefix).resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        include = Path(normalized) / "include"
        lib64 = Path(normalized) / "lib64"
        lib = Path(normalized) / "lib"
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
    description="Out-of-tree NCCL symmetric-windo w allocation for PyTorch 2.9+",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/nixnew657/torch-nccl-symm-extension",
    project_urls={
        "Source": "https://github.com/nixnew657/torch-nccl-symm-extension",
        "Issues": "https://github.com/nixnew657/torch-nccl-symm-extension/issues",
    },
    author="nccl-symm-mem contributors",
    license="MIT",
    license_files=["LICENSE"],
    keywords=["pytorch", "nccl", "cuda", "distributed", "symmetric-memory"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
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
            # CUMEM validation uses CUDA Driver API symbols (cuMemRetainAllocationHandle
            # and cuMemRelease), so link libcuda explicitly rather than relying on a
            # transitive dependency from PyTorch or NCCL.
            libraries=["nccl", "cuda", "dl"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.9",
    install_requires=["torch>=2.8"],
)
