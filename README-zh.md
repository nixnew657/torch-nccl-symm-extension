# NCCL Symmetric Memory Extension

一个面向 PyTorch 的独立扩展：通过 `ncclMemAlloc` 分配 tensor，并通过
`ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册到已有的
`ProcessGroupNCCL` communicator。注册后的 tensor 可直接传给标准
`torch.distributed` 集合通信接口。
可与 torch 2.9 等不支持symmetric的结合使用

## 构建

```
运行时 NCCL >= 2.27.3（rendezvous() 强制校验；建议 2.30.7+）
```

使用 `pip` 提供的 PEP 517 构建接口：

```bash
cd nccl-symm-mem-extension
NCCL_HOME=/path/to/nccl python3 -m pip install --no-build-isolation .
```

`--no-build-isolation` 会复用当前环境已安装的 PyTorch。扩展的构建过程依赖
`torch.utils.cpp_extension`；该选项可以避免 `pip` 在隔离构建环境中下载另一份
PyTorch wheel。

如需只构建可分发的 wheel、暂不安装：

```bash
NCCL_HOME=/path/to/nccl python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
```

该扩展仅调用 NCCL/CUDA host API，不会编译 CUDA device code。

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

`rendezvous()` 是子组内的 collective 操作：子组成员必须以相同顺序调用。
`registration.close()` 在各自 rank 上本地注销窗口，**不是** collective（NCCL
注销不做 barrier）。调用前，每个 rank 都必须确保没有尚未完成的集合通信在访问
该注册范围；销毁或中止 ProcessGroup 前，各 rank 仍应按相同顺序调用 `close()`。

`x` 可以是 `symm.empty()` 返回的完整 tensor，也可以是其中连续、字节偏移为
4096 整数倍的子视图（例如 float32 tensor 的 `x[1024:]`）。所有 rank 必须注册
相同的偏移与大小；其他视图会被拒绝。同一 communicator 内与已注册窗口重叠的
注册也会被拒绝。

同一 tensor（或相同的对齐子窗口）可同时注册到多个 communicator，例如 WORLD
和 subgroup；这些注册彼此独立，只有同一 communicator 中的窗口不能重叠。

`empty()` 会验证 `ncclMemAlloc` 的返回内存是否具备 CUDA CUMEM/VMM 支持；若不
支持，会提前给出可诊断的错误。当 communicator 不支持对称内存（无全
P2P/NVLink 或 GIN 拓扑，或 `NCCL_WIN_ENABLE=0`）时，NCCL 可能成功返回但不
创建窗口；`rendezvous()` 会检测该情况并抛出 `RuntimeError`，而不是静默退回
普通集合通信。

## 验证

```bash
cd nccl-symm-mem-extension
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  torchrun --standalone --nproc_per_node=4 tests/demo_distributed.py
```

满足 NCCL 对称内存拓扑条件时，日志应包含：

```text
AllReduce [Symmetric]: ...
```
