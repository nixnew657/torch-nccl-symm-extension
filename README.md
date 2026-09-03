# NCCL Symmetric Memory Extension

[中文文档](README-zh.md)

A standalone PyTorch extension that allocates tensors with `ncclMemAlloc` and
registers them with an existing `ProcessGroupNCCL` communicator through
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`. Once registered, these
tensors can be passed directly to standard `torch.distributed` collective APIs.

This enables NCCL symmetric-memory collectives when using PyTorch versions such
as PyTorch 2.9 that do not provide a native symmetric-memory allocation API.

## Requirements

- NCCL 2.30.7+ is recommended for the latest symmetric-memory kernels and fixes
- PyTorch with the NCCL backend
- CUDA-capable GPU(s)

## Build

Use the PEP 517 build interface exposed by `pip`:

```bash
cd torch-nccl-symm-extension
NCCL_HOME=/path/to/nccl python3 -m pip install --no-build-isolation .
```

`--no-build-isolation` makes the build use the installed PyTorch package. This
is required because `torch.utils.cpp_extension` is used to build the extension
and avoids `pip` downloading another PyTorch wheel into an isolated build
environment.

To create a redistributable wheel instead of installing it immediately:

```bash
NCCL_HOME=/path/to/nccl python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
```
without NCCL_HOME, the build will find site-package nvidia/nccl.

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

`rendezvous()` is a collective within the given process group: all members must
call it in the same order. `registration.close()` deregisters the window locally
on each rank and is **not** a collective (NCCL performs no barrier for
deregistration). Before calling it, every rank must have finished all in-flight
collectives accessing the registered range. Ranks should still call `close()` in
the same order before destroying or aborting the ProcessGroup.

A tensor (or the same aligned subwindow) may be registered concurrently with
multiple communicators, for example WORLD and a subgroup. Those registrations
are independent. Overlapping registrations are rejected only within one
communicator.

`x` must be the original tensor returned by `symm.empty()`, or a contiguous view
of it whose byte offset is a multiple of 4096 (e.g. `x[1024:]` for a float32
tensor). Every rank must register the same offset and size; other views are
rejected. Registering a window that overlaps one already registered in the same
group is also rejected.

`empty()` validates that its `ncclMemAlloc` result is backed by CUDA CUMEM/VMM
and fails early with an actionable error when it is not. On platforms where the
communicator does not support symmetric memory (no all-P2P/NVLink or GIN
topology, or `NCCL_WIN_ENABLE=0`), NCCL can return success with no window.
`rendezvous()` detects that condition and raises a `RuntimeError` instead of
silently falling back to regular collectives.

## Verification

```bash
cd torch-nccl-symm-extension
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  torchrun --standalone --nproc_per_node=4 tests/demo_distributed.py
```

When the NCCL topology meets the symmetric-memory requirements, the NCCL log
should include:

```text
AllReduce [Symmetric]: ...
```
