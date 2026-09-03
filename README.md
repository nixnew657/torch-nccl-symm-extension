# NCCL Symmetric Memory Extension

[中文文档](README-zh.md)

A standalone PyTorch extension that allocates tensors with `ncclMemAlloc` and
registers them with an existing `ProcessGroupNCCL` communicator through
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`. Once registered, these
tensors can be passed directly to standard `torch.distributed` collective APIs.

This enables NCCL symmetric-memory collectives when using PyTorch versions such
as PyTorch 2.9 that do not provide a native symmetric-memory allocation API.

## Requirements

- NCCL version 2.30.7 or later
- PyTorch with the NCCL backend
- CUDA-capable GPU(s)

## Build

```bash
cd nccl-symm-mem-extension
NCCL_HOME=/path/to/nccl python3 setup.py build_ext --inplace
```

The extension only invokes NCCL and CUDA host APIs, so it is built with
`CppExtension` and does not compile CUDA device code.

## Usage

```python
import os

import torch
import torch.distributed as dist
import nccl_symm_mem as symm

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl", device_id=device)

# Every WORLD rank must create the subgroup in the same order.
subgroup_ranks = [0, 2]
subgroup = dist.new_group(subgroup_ranks, backend="nccl", device_id=device)

if dist.get_rank() in subgroup_ranks:
    x = symm.empty(1024, dtype=torch.float32, device=device)
    registration = symm.rendezvous(x, group=subgroup)
    try:
        dist.all_reduce(x, group=subgroup)
    finally:
        registration.close()

dist.destroy_process_group()
```

Both `rendezvous()` and `registration.close()` are collectives within the given
process group: all members of that group must call them in the same order.
`x` must be the original, complete tensor returned by `symm.empty()`; tensor
views are not supported.

## Verification

```bash
cd nccl-symm-mem-extension
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  torchrun --standalone --nproc_per_node=4 tests/smoke_distributed.py
```

When the NCCL topology meets the symmetric-memory requirements, the NCCL log
should include:

```text
AllReduce [Symmetric]: ...
```
