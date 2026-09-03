import os

import pytest


torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

import nccl_symm_mem as symm


def _require_distributed_cuda() -> None:
    if not torch.cuda.is_available() or not dist.is_available():
        pytest.skip("requires CUDA and torch.distributed")
    if not dist.is_initialized():
        pytest.skip("run under torchrun with NCCL initialized")


def test_empty_tracks_full_allocations() -> None:
    _require_distributed_cuda()
    x = symm.empty(256, dtype=torch.float32, device=torch.cuda.current_device())
    assert x.is_cuda
    assert x.is_contiguous()
    assert symm.is_symmetric_tensor(x)
    assert not symm.is_symmetric_tensor(x[1:])


def test_rendezvous_and_standard_all_reduce() -> None:
    _require_distributed_cuda()
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


def test_rendezvous_and_standard_all_reduce_on_subgroup() -> None:
    _require_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("requires at least two torchrun ranks")

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


def test_rendezvous_rejects_regular_tensor() -> None:
    _require_distributed_cuda()
    x = torch.empty(32, device="cuda")
    with pytest.raises(ValueError, match="nccl_symm_mem.empty"):
        symm.rendezvous(x)


def test_close_is_not_idempotent() -> None:
    _require_distributed_cuda()
    x = symm.empty(64, device=torch.cuda.current_device())
    registration = symm.rendezvous(x)
    registration.close()
    with pytest.raises(RuntimeError, match="already closed"):
        registration.close()
