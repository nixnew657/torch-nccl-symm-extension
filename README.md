# NCCL Symmetric Memory Extension

一个面向 PyTorch 的独立扩展：通过 `ncclMemAlloc` 分配 tensor，并通过
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册到已有的
`ProcessGroupNCCL` communicator。注册后的 tensor 可直接传给标准
`torch.distributed` 集合通信接口。

## 构建

```bash
cd nccl-symm-mem-extension
NCCL_HOME=/path/to/nccl MAX_JOBS=1 python3 setup.py build_ext --inplace
```

该扩展仅调用 NCCL/CUDA host API，不包含 CUDA kernel

## 使用

```python
import os

import torch
import torch.distributed as dist
import nccl_symm_mem as symm

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl", device_id=device)

# 所有 WORLD rank 都必须以相同顺序创建子组。
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

`rendezvous()` 和 `registration.close()` 都是子组内的 collective 操作：子组成员必须以相同顺序调用。`x` 必须是 `symm.empty()` 返回的原始完整 tensor，不能是 view。

## 验证

```bash

cd nccl-symm-mem-extension
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  torchrun --standalone --nproc_per_node=4 tests/smoke_distributed.py
```

满足 NCCL 对称内存拓扑条件时，日志应包含：

```text
AllReduce [Symmetric]: ...
```
