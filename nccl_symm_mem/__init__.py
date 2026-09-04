from __future__ import annotations

import inspect
from collections.abc import Sequence
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist

from . import _C


_registrations: dict[tuple[int, int], SymmetricRegistration] = {}
_registrations_lock = RLock()
_mem_pools: dict[torch.device, torch.cuda.MemPool] = {}


class SymmetricRegistration:
    """A view handle over a cached ``(allocation, NCCL communicator)`` window.

    Window registration is collective and happens once for the whole allocation.
    Views of that allocation retain their byte offset in independent handles. The
    NCCL window remains alive until the allocation itself is released, matching
    PyTorch's native symmetric-memory lifecycle. ``close()`` is intentionally
    idempotent and only releases this Python view handle.
    """

    def __init__(
        self,
        registration: Any,
        tensor: torch.Tensor,
        group_key: str,
        registry_key: tuple[int, int],
    ) -> None:
        self._registration = registration
        self._tensor: torch.Tensor | None = tensor
        self._group_key = group_key
        self._registry_key = registry_key

    def _require_open(self) -> Any:
        if self.closed:
            raise RuntimeError("symmetric registration is closed")
        return self._registration

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("symmetric registration is closed")
        return self._tensor

    @property
    def closed(self) -> bool:
        return bool(self._registration.closed)

    @property
    def nbytes(self) -> int:
        return int(self._registration.nbytes)

    @property
    def buffer_size(self) -> int:
        return int(self._registration.buffer_size)

    @property
    def offset(self) -> int:
        return int(self._registration.offset)

    @property
    def signal_pad_size(self) -> int:
        return int(self._registration.signal_pad_size)

    @property
    def rank(self) -> int:
        return int(self._registration.rank)

    @property
    def world_size(self) -> int:
        return int(self._registration.world_size)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", int(self._registration.device_index))

    @property
    def group_key(self) -> str:
        return self._group_key

    @property
    def peer_buffer_ptrs(self) -> list[int]:
        return [int(ptr) for ptr in self._require_open().peer_buffer_ptrs]

    @property
    def multicast_ptr(self) -> int:
        return int(self._require_open().multicast_ptr)

    def close(self) -> None:
        """Release this view handle without deregistering the shared NCCL window."""
        if self.closed:
            return
        self._registration.close()
        self._tensor = None
        with _registrations_lock:
            _registrations.pop(self._registry_key, None)

    def get_buffer(
        self,
        peer: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a tensor view of ``peer``'s buffer at this handle's offset."""
        return self._require_open().get_buffer(peer, tuple(sizes), dtype, storage_offset)

    def get_remote_tensor(
        self, peer: int, sizes: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        return self.get_buffer(peer, sizes, dtype)

    def get_signal_pad(
        self,
        peer: int,
        sizes: Sequence[int] = (),
        dtype: torch.dtype = torch.uint32,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        return self._require_open().get_signal_pad(peer, tuple(sizes), dtype, storage_offset)

    def has_peer_access(self, peer: int) -> bool:
        return bool(self._require_open().has_peer_access(peer))

    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
        self._require_open().barrier(channel, timeout_ms)

    def put_signal(self, peer: int, channel: int = 0, timeout_ms: int = 0) -> None:
        self._require_open().put_signal(peer, channel, timeout_ms)

    def wait_signal(self, peer: int, channel: int = 0, timeout_ms: int = 0) -> None:
        self._require_open().wait_signal(peer, channel, timeout_ms)

    def get_peer_cft_handle(self, peer: int) -> tuple[int, int]:
        le_id, le_offset = self._require_open().get_peer_cft_handle(peer)
        return int(le_id), int(le_offset)

    def get_multimem_cft_handle(self) -> tuple[int, int]:
        le_id, le_offset = self._require_open().get_multimem_cft_handle()
        return int(le_id), int(le_offset)

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
    alloc_id: int | None = None,
) -> torch.Tensor:
    """Allocate a CUDA tensor backed by NCCL VMM symmetric memory.

    An ``alloc_id`` requests persistent allocation: after the previous tensor
    dies, a subsequent identical request reuses the same address. This is useful
    when a compiled communication plan needs stable symmetric-memory addresses.
    """
    if len(size) == 1 and isinstance(size[0], Sequence):
        shape = tuple(int(dim) for dim in size[0])
    else:
        shape = tuple(int(dim) for dim in size)
    if alloc_id is not None and alloc_id < 0:
        raise ValueError("alloc_id must be non-negative")
    return _C.empty(shape, dtype, _normalize_device(device), alloc_id)


def track_tensor(tensor: torch.Tensor) -> None:
    """Register a contiguous tensor from a symmetric NCCL ``MemPool`` for rendezvous."""
    _C.track_tensor(tensor)


def release_persistent_allocation(alloc_id: int) -> None:
    """Release an inactive persistent allocation before CUDA teardown.

    The tensor returned by :func:`empty` with the same ``alloc_id`` must already
    be destroyed. Call this explicitly during shutdown to avoid deferring NCCL
    memory release until Python interpreter finalization.
    """
    if alloc_id < 0:
        raise ValueError("alloc_id must be non-negative")
    _C.release_persistent_allocation(int(alloc_id))


def get_mem_pool(device: torch.device | str | int | None = None) -> torch.cuda.MemPool:
    """Return an NCCL allocator pool suitable for tracked symmetric allocations.

    PyTorch 2.9 does not expose ``MemPool.no_split``; newer releases do. Use
    the stronger no-split setting when it is available without excluding 2.9.
    The caller must allocate equivalent tensors in the same order on every rank,
    call :func:`track_tensor` for a tensor before rendezvous, and retain the pool
    until every associated tensor/window has been released.
    """
    normalized = _normalize_device(device)
    if normalized not in _mem_pools:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating a NCCL MemPool")
        group = dist.group.WORLD
        backend = group._get_backend(normalized)
        allocator = getattr(backend, "mem_allocator", None)
        if allocator is None:
            raise RuntimeError("the selected NCCL backend does not expose mem_allocator")
        mem_pool_kwargs: dict[str, bool] = {"use_on_oom": False}
        if "no_split" in inspect.signature(torch.cuda.MemPool).parameters:
            mem_pool_kwargs["no_split"] = True
        _mem_pools[normalized] = torch.cuda.MemPool(allocator, **mem_pool_kwargs)
    return _mem_pools[normalized]


def rendezvous(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> SymmetricRegistration:
    """Collectively establish symmetric access for ``tensor`` within ``group``.

    The full underlying allocation is registered once per NCCL communicator.
    Contiguous subviews are accepted without a page-alignment constraint; their
    handle retains the view byte offset. Repeated calls for the same view and
    communicator return a cached view handle.
    """
    capabilities = nccl_capabilities()
    if not capabilities["windows"]:
        raise RuntimeError(
            f"runtime NCCL version {nccl_version()} is older than 2.27.0, the "
            "minimum version supporting ncclCommWindowRegister"
        )
    if not tensor.is_cuda:
        raise ValueError("tensor must be CUDA")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if not is_symmetric_tensor(tensor):
        raise ValueError(
            "tensor must be nccl_symm_mem.empty() output, a view fully contained "
            "within one of those allocations, or a tracked symmetric MemPool tensor"
        )

    resolved_group = _resolve_group(group)
    device = _normalize_device(tensor.device)
    comm_ptr = _get_comm_ptr(resolved_group, device)
    group_key = _group_key(resolved_group)
    registry_key = (tensor.data_ptr(), comm_ptr)
    with _registrations_lock:
        cached = _registrations.get(registry_key)
        if cached is not None and not cached.closed:
            return cached
        registration = _C._SymmetricRegistration(
            tensor,
            comm_ptr,
            group_key,
            dist.get_rank(resolved_group),
            dist.get_world_size(resolved_group),
        )
        handle = SymmetricRegistration(registration, tensor, group_key, registry_key)
        _registrations[registry_key] = handle
        return handle


def is_symmetric_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is contiguous and belongs to tracked symmetric memory."""
    return bool(_C.is_symmetric_tensor(tensor))


def nccl_version() -> int:
    """Return the dynamically linked NCCL version as an encoded integer."""
    return int(_C.nccl_version())


def nccl_capabilities() -> dict[str, bool | int]:
    """Return runtime gates plus NCCL/CUDA header versions used to build the extension."""
    return {
        str(name): (int(value) if name.endswith("_version") else bool(value))
        for name, value in _C.capabilities().items()
    }


def get_signal_pad_size() -> int:
    return int(_C.get_signal_pad_size())


def set_signal_pad_size(size: int) -> None:
    _C.set_signal_pad_size(int(size))


def get(dst: torch.Tensor, hdl: SymmetricRegistration, peer: int, offset: int = 0) -> None:
    """Copy from a peer's symmetric buffer into a contiguous local tensor."""
    if dst.device != hdl.device:
        raise ValueError("get: dst must be on the same device as hdl")
    if not dst.is_contiguous():
        raise ValueError("get: dst must be contiguous")
    if offset < 0:
        raise ValueError("get: offset must be non-negative")
    remote = hdl.get_buffer(peer, (dst.numel(),), dst.dtype, offset)
    dst.copy_(remote)


__all__ = [
    "SymmetricRegistration",
    "empty",
    "get",
    "get_mem_pool",
    "get_signal_pad_size",
    "is_symmetric_tensor",
    "nccl_capabilities",
    "nccl_version",
    "rendezvous",
    "release_persistent_allocation",
    "set_signal_pad_size",
    "track_tensor",
]
