from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist

from . import _C


_registrations: dict[tuple[int, str], SymmetricRegistration] = {}
_registrations_lock = RLock()


class SymmetricRegistration:
    """Own one collective NCCL symmetric-window registration.

    Keep this object alive until all collectives using ``tensor`` have completed.
    ``close()`` is collective: every process group rank must call it in the same
    order before the ProcessGroup is destroyed or aborted.
    """

    def __init__(
        self,
        registration: Any,
        tensor: torch.Tensor,
        group_key: str,
        registry_key: tuple[int, str],
    ) -> None:
        self._registration = registration
        self._tensor: torch.Tensor | None = tensor
        self._group_key = group_key
        self._registry_key = registry_key

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("symmetric registration is closed")
        return self._tensor

    @property
    def closed(self) -> bool:
        return self._registration.closed

    @property
    def nbytes(self) -> int:
        return self._registration.nbytes

    @property
    def group_key(self) -> str:
        return self._group_key

    def close(self) -> None:
        """Collectively deregister the tensor's NCCL symmetric window."""
        self._registration.close()
        self._tensor = None
        with _registrations_lock:
            _registrations.pop(self._registry_key, None)

    def __enter__(self) -> SymmetricRegistration:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _normalize_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NCCL symmetric memory")
        return torch.device("cuda", torch.cuda.current_device())
    normalized = torch.device(device)
    if normalized.type != "cuda":
        raise ValueError(f"device must be CUDA, got {normalized}")
    if normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def _resolve_group(group: dist.ProcessGroup | None) -> dist.ProcessGroup:
    if group is None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before rendezvous")
        return dist.group.WORLD
    return group


def _group_key(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    if isinstance(name, str) and name:
        return name
    return f"process_group:{id(group)}"


def _get_comm_ptr(group: dist.ProcessGroup, device: torch.device) -> int:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before rendezvous")
    backend_name = str(dist.get_backend(group)).lower()
    if backend_name != "nccl":
        raise ValueError(f"group must use the NCCL backend, got {backend_name!r}")

    backend = group._get_backend(device)
    comm_ptr_method = getattr(backend, "_comm_ptr", None)
    if comm_ptr_method is None:
        raise RuntimeError(
            "the selected NCCL backend does not expose _comm_ptr(); "
            "this extension requires PyTorch 2.9 ProcessGroupNCCL"
        )
    comm_ptr = int(comm_ptr_method())
    if comm_ptr == 0:
        raise RuntimeError(
            "the NCCL communicator has not been initialized for this device. "
            "Pass device_id=device to dist.init_process_group(), or run one "
            "NCCL collective on this device before rendezvous."
        )
    return comm_ptr


def empty(
    *size: int | Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Allocate a CUDA tensor with ``ncclMemAlloc``.

    This function only allocates. Call :func:`rendezvous` collectively before
    passing the tensor to ``torch.distributed`` collectives.
    """
    if len(size) == 1 and isinstance(size[0], Sequence):
        shape = tuple(int(dim) for dim in size[0])
    else:
        shape = tuple(int(dim) for dim in size)
    if not shape:
        raise ValueError("at least one dimension is required")
    return _C.empty(shape, dtype, _normalize_device(device))


def rendezvous(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> SymmetricRegistration:
    """Collectively register ``tensor`` as an NCCL symmetric window.

    Every participating rank must call this function in identical order with a
    full allocation of the same byte size. After it returns, standard calls such
    as ``dist.all_reduce(tensor, group=group)`` use the same ProcessGroup NCCL
    communicator and are eligible for NCCL symmetric-memory kernels.
    """
    if not _C.supports_symmetric_windows():
        raise RuntimeError("this extension was built against NCCL < 2.27")
    if not tensor.is_cuda:
        raise ValueError("tensor must be CUDA")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if not _C.is_symmetric_tensor(tensor):
        raise ValueError(
            "tensor must be the full, original tensor returned by nccl_symm_mem.empty()"
        )

    resolved_group = _resolve_group(group)
    device = _normalize_device(tensor.device)
    comm_ptr = _get_comm_ptr(resolved_group, device)
    key = _group_key(resolved_group)
    registry_key = (tensor.data_ptr(), key)
    with _registrations_lock:
        cached = _registrations.get(registry_key)
        if cached is not None and not cached.closed:
            return cached
        registration = _C._SymmetricRegistration(tensor, comm_ptr, key)
        handle = SymmetricRegistration(registration, tensor, key, registry_key)
        _registrations[registry_key] = handle
        return handle


def is_symmetric_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a full allocation created by :func:`empty`."""
    return bool(_C.is_symmetric_tensor(tensor))


def nccl_version() -> int:
    """Return the NCCL version the extension is linked against, as an integer."""
    return int(_C.nccl_version())


__all__ = [
    "SymmetricRegistration",
    "empty",
    "is_symmetric_tensor",
    "nccl_version",
    "rendezvous",
]
