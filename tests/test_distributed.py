import os
from functools import wraps

import pytest


torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

import nccl_symm_mem as symm


@pytest.fixture(scope="session", autouse=True)
def _initialize_nccl_process_group() -> None:
    """Initialize the per-rank NCCL group when pytest is launched by torchrun."""
    if not torch.cuda.is_available() or not dist.is_available():
        yield
        return
    if dist.is_initialized():
        yield
        return
    if "LOCAL_RANK" not in os.environ:
        yield
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        yield
    finally:
        # ProcessGroupNCCL owns asynchronous work; wait before tearing down its
        # communicator after all tests on this rank have finished.
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def _nccl_version_code(version: tuple[int, int, int]) -> int:
    major, minor, patch = version
    return major * 10_000 + minor * 100 + patch


def requires_distributed_cuda(
    *,
    min_cuda_devices: int = 1,
    min_world_size: int = 1,
    min_nccl_version: tuple[int, int, int] | None = None,
):
    """Skip a test unless its CUDA, NCCL, and torchrun requirements are met."""
    def decorator(test):
        @wraps(test)
        def wrapped(*args, **kwargs):
            if not torch.cuda.is_available() or not dist.is_available():
                pytest.skip("requires CUDA and torch.distributed")
            if not dist.is_nccl_available():
                pytest.skip("requires a PyTorch build with the NCCL backend")
            if torch.cuda.device_count() < min_cuda_devices:
                pytest.skip(
                    f"requires at least {min_cuda_devices} CUDA devices, found "
                    f"{torch.cuda.device_count()}"
                )
            if not dist.is_initialized():
                pytest.skip("run under torchrun, which initializes NCCL for this test suite")
            if dist.get_backend() != "nccl":
                pytest.skip(f"requires the NCCL backend, found {dist.get_backend()}")
            if dist.get_world_size() < min_world_size:
                pytest.skip(
                    f"requires at least {min_world_size} torchrun ranks, found "
                    f"{dist.get_world_size()}"
                )
            if min_nccl_version is not None:
                required_version = _nccl_version_code(min_nccl_version)
                actual_version = symm.nccl_version()
                if actual_version < required_version:
                    pytest.skip(
                        f"requires NCCL >= {min_nccl_version}, found "
                        f"{actual_version // 10_000}."
                        f"{actual_version % 10_000 // 100}."
                        f"{actual_version % 100}"
                    )
            return test(*args, **kwargs)

        return wrapped

    return decorator


_NCCL_WINDOW_VERSION = (2, 27, 0)


def test_cft_compile_time_capability_matches_header_versions() -> None:
    capabilities = symm.nccl_capabilities()
    nccl_headers_support_cft = capabilities["nccl_header_version"] >= 23102
    cuda_headers_support_cft = capabilities["cuda_header_version"] >= 13030
    assert capabilities["host_cft_build"] is (
        nccl_headers_support_cft and cuda_headers_support_cft
    )
    assert not capabilities["host_cft"] or capabilities["host_cft_build"]


@requires_distributed_cuda()
def test_empty_tracks_full_allocations() -> None:
    x = symm.empty(4096, dtype=torch.float32, device=torch.cuda.current_device())
    assert x.is_cuda
    assert x.is_contiguous()
    assert symm.is_symmetric_tensor(x)
    # All contiguous views are recognized and may rendezvous; the shared
    # allocation window is page-aligned, while a handle retains each view offset.
    assert symm.is_symmetric_tensor(x[1:])
    assert symm.is_symmetric_tensor(x[1024:])
    assert not symm.is_symmetric_tensor(x[::2])  # non-contiguous view


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_rendezvous_and_standard_all_reduce() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    x = symm.empty(1024, dtype=torch.float32, device=device)
    x.fill_(rank + 1)
    registration = symm.rendezvous(x)
    try:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(x, torch.full_like(x, expected))
        assert not registration.closed
    finally:
        registration.close()
    assert registration.closed


@requires_distributed_cuda(min_world_size=2, min_nccl_version=_NCCL_WINDOW_VERSION)
def test_rendezvous_and_standard_all_reduce_on_subgroup() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    subgroup_ranks = list(range(0, world_size, 2))
    subgroup = dist.new_group(ranks=subgroup_ranks, backend="nccl", device_id=device)
    try:
        if rank not in subgroup_ranks:
            return
        x = symm.empty(1024, dtype=torch.float32, device=device)
        x.fill_(rank + 1)
        registration = symm.rendezvous(x, group=subgroup)
        try:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=subgroup)
            torch.cuda.synchronize(device)
            expected = sum(member_rank + 1 for member_rank in subgroup_ranks)
            torch.testing.assert_close(x, torch.full_like(x, expected))
        finally:
            registration.close()
    finally:
        dist.destroy_process_group(subgroup)


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_rendezvous_rejects_regular_tensor() -> None:
    x = torch.empty(32, device="cuda")
    with pytest.raises(ValueError, match="nccl_symm_mem.empty"):
        symm.rendezvous(x)


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_close_is_idempotent() -> None:
    x = symm.empty(64, device=torch.cuda.current_device())
    registration = symm.rendezvous(x)
    registration.close()
    registration.close()
    assert registration.closed


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_registration_metadata_and_cached_view_handle() -> None:
    x = symm.empty(1024, dtype=torch.float32, device=torch.cuda.current_device())
    view = x[1:]
    first = symm.rendezvous(view)
    second = symm.rendezvous(view)
    try:
        assert second is first
        assert first.offset == x.element_size()
        assert first.buffer_size == x.numel() * x.element_size()
        assert first.nbytes == view.numel() * view.element_size()
        assert first.rank == dist.get_rank()
        assert first.world_size == dist.get_world_size()
        assert first.signal_pad_size >= 4096
        assert len(first.peer_buffer_ptrs) == dist.get_world_size()
    finally:
        first.close()


@requires_distributed_cuda()
def test_persistent_allocation_reuses_address_after_release() -> None:
    alloc_id = 123456
    device = torch.device("cuda", torch.cuda.current_device())
    first = symm.empty(256, dtype=torch.float32, device=device, alloc_id=alloc_id)
    first_ptr = first.data_ptr()
    second = None
    try:
        with pytest.raises(RuntimeError, match="still active"):
            symm.empty(256, dtype=torch.float32, device=device, alloc_id=alloc_id)
        del first
        second = symm.empty(256, dtype=torch.float32, device=device, alloc_id=alloc_id)
        assert second.data_ptr() == first_ptr
    finally:
        if second is not None:
            del second
        elif "first" in locals():
            del first
        symm.release_persistent_allocation(alloc_id)


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_rendezvous_subwindow_with_equal_offset() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    x = symm.empty(2048, dtype=torch.float32, device=device)
    view = x[1024:]  # byte offset 4096 on every rank
    view.fill_(rank + 1)
    registration = symm.rendezvous(view)
    try:
        dist.all_reduce(view)
        torch.cuda.synchronize(device)
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(view, torch.full_like(view, expected))
        assert not registration.closed
    finally:
        registration.close()


@requires_distributed_cuda(min_nccl_version=_NCCL_WINDOW_VERSION)
def test_rendezvous_accepts_misaligned_and_overlapping_views() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    x = symm.empty(4096, dtype=torch.float32, device=device)
    misaligned = x[1:]
    overlapping = x[2:]
    first = symm.rendezvous(misaligned)
    second = symm.rendezvous(overlapping)
    try:
        assert first.offset == x.element_size()
        assert second.offset == 2 * x.element_size()
        misaligned.fill_(rank + 1)
        dist.all_reduce(misaligned)
        torch.cuda.synchronize(device)
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(misaligned, torch.full_like(misaligned, expected))
    finally:
        first.close()
        second.close()


@requires_distributed_cuda(min_world_size=2, min_nccl_version=_NCCL_WINDOW_VERSION)
def test_peer_buffer_and_signal_interfaces() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    peer = (rank + 1) % dist.get_world_size()
    x = symm.empty(1024, dtype=torch.float32, device=device)
    x.fill_(rank)
    registration = symm.rendezvous(x)
    try:
        assert registration.get_buffer(rank, x.shape, x.dtype).data_ptr() == x.data_ptr()
        assert registration.get_signal_pad(rank).numel() == registration.signal_pad_size // 4
        if registration.has_peer_access(peer):
            remote = registration.get_buffer(peer, x.shape, x.dtype)
            assert remote.is_cuda
            dist.barrier()
            torch.testing.assert_close(remote, torch.full_like(remote, peer))
            dist.barrier()
    finally:
        registration.close()


@requires_distributed_cuda(min_world_size=2, min_nccl_version=_NCCL_WINDOW_VERSION)
def test_same_tensor_can_be_registered_by_world_and_subgroup() -> None:
    """Windows on separate NCCL communicators are independent and may overlap."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    subgroup_ranks = list(range(0, world_size, 2))
    subgroup = dist.new_group(ranks=subgroup_ranks, backend="nccl", device_id=device)
    world_reg = None
    subgroup_reg = None
    try:
        x = symm.empty(1024, dtype=torch.float32, device=device)
        # Keep both registrations live simultaneously: their byte ranges are
        # identical, but their NCCL communicators are different.
        world_reg = symm.rendezvous(x, group=dist.group.WORLD)
        if rank in subgroup_ranks:
            subgroup_reg = symm.rendezvous(x, group=subgroup)
            assert subgroup_reg is not world_reg
        dist.barrier()

        x.fill_(rank + 1)
        dist.all_reduce(x, group=dist.group.WORLD)
        torch.cuda.synchronize(device)
        world_expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(x, torch.full_like(x, world_expected))
        dist.barrier()

        if rank in subgroup_ranks:
            x.fill_(rank + 1)
            dist.all_reduce(x, group=subgroup)
            torch.cuda.synchronize(device)
            subgroup_expected = sum(member + 1 for member in subgroup_ranks)
            torch.testing.assert_close(x, torch.full_like(x, subgroup_expected))
        dist.barrier()
    finally:
        # close() is local, but barriers above prove no collective still uses x.
        if subgroup_reg is not None:
            subgroup_reg.close()
        dist.barrier()
        if world_reg is not None:
            world_reg.close()
        dist.destroy_process_group(subgroup)
