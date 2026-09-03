import os

import torch
import torch.distributed as dist

import nccl_symm_mem as symm


def run_all_reduce(
    group: dist.ProcessGroup,
    group_ranks: list[int],
    device: torch.device,
    label: str,
) -> None:
    rank = dist.get_rank()
    tensor = symm.empty(1024, dtype=torch.float32, device=device)
    tensor.fill_(rank + 1)
    registration = symm.rendezvous(tensor, group=group)
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize(device)
        expected = sum(member_rank + 1 for member_rank in group_ranks)
        torch.testing.assert_close(tensor, torch.full_like(tensor, expected))
        print(
            f"rank={rank} {label}_ok ranks={group_ranks} "
            f"nbytes={registration.nbytes} nccl_version={symm.nccl_version()}",
            flush=True,
        )
    finally:
        registration.close()


def run_all_reduce_subwindow(
    group: dist.ProcessGroup,
    group_ranks: list[int],
    device: torch.device,
    label: str,
) -> None:
    rank = dist.get_rank()
    tensor = symm.empty(2048, dtype=torch.float32, device=device)
    view = tensor[1024:]  # byte offset 4096 on every rank
    view.fill_(rank + 1)
    registration = symm.rendezvous(view, group=group)
    try:
        dist.all_reduce(view, group=group)
        torch.cuda.synchronize(device)
        expected = sum(member_rank + 1 for member_rank in group_ranks)
        torch.testing.assert_close(view, torch.full_like(view, expected))
        print(
            f"rank={rank} {label}_ok ranks={group_ranks} "
            f"nbytes={registration.nbytes} nccl_version={symm.nccl_version()}",
            flush=True,
        )
    finally:
        registration.close()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    subgroup = None
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        run_all_reduce(dist.group.WORLD, list(range(world_size)), device, "world_all_reduce")
        run_all_reduce_subwindow(
            dist.group.WORLD, list(range(world_size)), device, "world_all_reduce_subwindow"
        )

        if world_size >= 2:
            subgroup_ranks = list(range(0, world_size, 2))
            subgroup = dist.new_group(ranks=subgroup_ranks, backend="nccl", device_id=device)
            if rank in subgroup_ranks:
                run_all_reduce(subgroup, subgroup_ranks, device, "subgroup_all_reduce")
                run_all_reduce_subwindow(
                    subgroup, subgroup_ranks, device, "subgroup_all_reduce_subwindow"
                )
        else:
            print("subgroup test skipped: requires at least two ranks", flush=True)
    finally:
        if subgroup is not None:
            dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
