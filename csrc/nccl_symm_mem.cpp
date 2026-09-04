#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

// NCCL's CFT implementation compiles its logical-endpoint runtime only when
// built against CUDA 13.3+ headers. Host-side CFT LE query APIs were added in
// NCCL 2.31.2. Keep this compile-time gate separate from runtime probing: it
// describes this extension's headers, not the CUDA version used to build the
// dynamically loaded libnccl.so.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 2) && CUDA_VERSION >= 13030
#define NCCL_SYMM_MEM_HAS_HOST_CFT_BUILD 1
#else
#define NCCL_SYMM_MEM_HAS_HOST_CFT_BUILD 0
#endif

namespace {

constexpr size_t kDefaultSignalPadSize = NCCL_WIN_REQUIRED_ALIGNMENT;
constexpr size_t kSignalPadAlignment = NCCL_WIN_REQUIRED_ALIGNMENT;

void check_nccl(ncclResult_t result, const char* operation) {
  TORCH_CHECK(result == ncclSuccess, operation, " failed: ", ncclGetErrorString(result));
}

size_t round_up(size_t value, size_t alignment) {
  TORCH_CHECK(alignment != 0, "alignment must be non-zero");
  TORCH_CHECK(value <= SIZE_MAX - (alignment - 1), "symmetric allocation size overflow");
  return ((value + alignment - 1) / alignment) * alignment;
}

int runtime_nccl_version() {
  int version = 0;
  check_nccl(ncclGetVersion(&version), "ncclGetVersion");
  return version;
}

template <typename Function>
Function lookup_nccl_symbol(const char* name) {
  return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

using GetPeerDevicePointerFn = ncclResult_t (*)(ncclWindow_t, size_t, int, void**);
using GetMultimemDevicePointerFn = ncclResult_t (*)(ncclWindow_t, size_t, void**);
using GetPeerCftInfoFn = ncclResult_t (*)(ncclWindow_t, size_t, int, uint32_t*, size_t*);
using GetMultimemCftInfoFn = ncclResult_t (*)(ncclWindow_t, size_t, uint32_t*, size_t*);

// Signal APIs are optional at runtime as well. Keep local ABI-compatible
// definitions so an extension built with newer NCCL headers remains importable
// when the dynamic loader resolves an older libnccl.so.
struct NcclWaitSignalDesc {
  int opCnt;
  int peer;
  int sigIdx;
  int ctx;
};
using SignalFn = ncclResult_t (*)(int, int, int, unsigned int, ncclComm_t, cudaStream_t);
using WaitSignalFn = ncclResult_t (*)(int, NcclWaitSignalDesc*, ncclComm_t, cudaStream_t);

GetPeerDevicePointerFn get_peer_device_pointer_fn() {
  return lookup_nccl_symbol<GetPeerDevicePointerFn>("ncclGetPeerDevicePointer");
}

GetMultimemDevicePointerFn get_multimem_device_pointer_fn() {
  return lookup_nccl_symbol<GetMultimemDevicePointerFn>("ncclGetLsaMultimemDevicePointer");
}

GetPeerCftInfoFn get_peer_cft_info_fn() {
  return lookup_nccl_symbol<GetPeerCftInfoFn>("ncclGetPeerDeviceLeInfo");
}

GetMultimemCftInfoFn get_multimem_cft_info_fn() {
  return lookup_nccl_symbol<GetMultimemCftInfoFn>("ncclGetMultimemDeviceLeInfo");
}

SignalFn get_signal_fn() {
  return lookup_nccl_symbol<SignalFn>("ncclSignal");
}

WaitSignalFn get_wait_signal_fn() {
  return lookup_nccl_symbol<WaitSignalFn>("ncclWaitSignal");
}

struct WindowState;

struct Allocation {
  Allocation(int device_index, size_t user_size, size_t signal_pad_size)
      : device_index(device_index),
        user_size(user_size),
        signal_pad_size(signal_pad_size),
        registration_size(signal_pad_size + round_up(user_size, 16)),
        owns_memory(true) {
    c10::cuda::CUDAGuard device_guard(device_index);
    check_nccl(ncclMemAlloc(&base_ptr, registration_size), "ncclMemAlloc");
    validate_vmm(base_ptr);
    if (signal_pad_size != 0) {
      TORCH_CHECK(
          cudaMemset(base_ptr, 0, signal_pad_size) == cudaSuccess,
          "cudaMemset(signal pad) failed");
    }
    data_ptr = static_cast<void*>(static_cast<char*>(base_ptr) + signal_pad_size);
  }

  Allocation(
      int device_index,
      void* base_ptr,
      size_t registration_size,
      size_t user_size)
      : device_index(device_index),
        base_ptr(base_ptr),
        data_ptr(base_ptr),
        user_size(user_size),
        registration_size(registration_size),
        owns_memory(false) {}

  ~Allocation();

  static void validate_vmm(void* ptr) {
    CUmemGenericAllocationHandle handle{};
    const CUresult retain_result = cuMemRetainAllocationHandle(&handle, ptr);
    TORCH_CHECK(
        retain_result == CUDA_SUCCESS,
        "the allocation is not backed by CUDA Driver VMM (cuMem*) APIs; NCCL "
        "symmetric memory requires NCCL's CUMEM path. Check CUDA Driver VMM "
        "support and ensure NCCL_CUMEM_ENABLE is not 0.");
    const CUresult release_result = cuMemRelease(handle);
    TORCH_CHECK(
        release_result == CUDA_SUCCESS,
        "failed to validate the CUDA Driver VMM allocation used by NCCL symmetric memory");
  }

  int device_index;
  void* base_ptr{nullptr};
  void* data_ptr{nullptr};
  size_t user_size{0};
  size_t signal_pad_size{0};
  size_t registration_size{0};
  bool owns_memory{false};
  std::mutex mutex;
  std::unordered_map<uintptr_t, std::shared_ptr<WindowState>> windows;
};

struct WindowState {
  WindowState(
      std::shared_ptr<Allocation> allocation,
      int64_t comm_ptr,
      std::string group_key,
      int rank,
      int world_size)
      : allocation_(allocation),
        comm_(reinterpret_cast<ncclComm_t>(comm_ptr)),
        group_key_(std::move(group_key)),
        rank_(rank),
        world_size_(world_size),
        device_index_(allocation->device_index),
        peer_window_bases_(static_cast<size_t>(world_size), nullptr) {
    TORCH_CHECK(comm_ != nullptr, "NCCL communicator is not initialized");
    TORCH_CHECK(!group_key_.empty(), "group_key must not be empty");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "invalid process group rank/world size");
    const auto owned_allocation = allocation;
    c10::cuda::CUDAGuard device_guard(owned_allocation->device_index);
    check_nccl(
        ncclCommWindowRegister(
            comm_,
            owned_allocation->base_ptr,
            owned_allocation->registration_size,
            &window_,
            NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)");
    TORCH_CHECK(
        window_ != nullptr,
        "the NCCL communicator does not support symmetric memory; window registration "
        "was accepted without creating a window. Ensure the topology supports symmetric "
        "memory, CUMEM is enabled, and NCCL_WIN_ENABLE is not disabled.");
    peer_window_bases_[static_cast<size_t>(rank_)] = owned_allocation->base_ptr;
  }

  ~WindowState() {
    if (window_ == nullptr || comm_ == nullptr) {
      return;
    }
    cudaSetDevice(device_index_);
    ncclCommWindowDeregister(comm_, window_);
  }

  void* peer_window_base(int peer) {
    TORCH_CHECK(peer >= 0 && peer < world_size_, "peer rank is out of range");
    std::lock_guard<std::mutex> lock(peer_mutex_);
    auto& peer_ptr = peer_window_bases_[static_cast<size_t>(peer)];
    if (peer_ptr != nullptr) {
      return peer_ptr;
    }
    const auto fn = get_peer_device_pointer_fn();
    TORCH_CHECK(
        fn != nullptr,
        "peer buffer discovery requires NCCL >= 2.29 with ncclGetPeerDevicePointer; "
        "the runtime NCCL library does not expose it");
    c10::cuda::CUDAGuard device_guard(allocation()->device_index);
    check_nccl(fn(window_, 0, peer, &peer_ptr), "ncclGetPeerDevicePointer");
    TORCH_CHECK(
        peer_ptr != nullptr,
        "peer symmetric memory is not directly accessible; the peer is outside the "
        "NCCL LSA/NVLink domain");
    return peer_ptr;
  }

  void* multicast_data_ptr() {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    if (multicast_data_ptr_ != nullptr) {
      return multicast_data_ptr_;
    }
    const auto fn = get_multimem_device_pointer_fn();
    TORCH_CHECK(
        fn != nullptr,
        "multicast symmetric memory requires NCCL >= 2.29 with "
        "ncclGetLsaMultimemDevicePointer. Ensure the runtime NCCL library "
        "provides this optional symbol.");
    const auto owned_allocation = allocation();
    c10::cuda::CUDAGuard device_guard(owned_allocation->device_index);
    void* ptr = nullptr;
    const auto result = fn(window_, owned_allocation->signal_pad_size, &ptr);
    TORCH_CHECK(
        result == ncclSuccess && ptr != nullptr,
        "NCCL multicast (NVLS/multimem) is unavailable for this communicator: ",
        ncclGetErrorString(result));
    multicast_data_ptr_ = ptr;
    return multicast_data_ptr_;
  }

  std::weak_ptr<Allocation> allocation_;

  std::shared_ptr<Allocation> allocation() const {
    auto result = allocation_.lock();
    TORCH_CHECK(result != nullptr, "symmetric allocation has already been released");
    return result;
  }

  ncclComm_t comm_{nullptr};
  std::string group_key_;
  int rank_{0};
  int world_size_{0};
  int device_index_{-1};
  ncclWindow_t window_{nullptr};
  std::mutex peer_mutex_;
  std::vector<void*> peer_window_bases_;
  void* multicast_data_ptr_{nullptr};
};

Allocation::~Allocation() {
  windows.clear();
  if (!owns_memory || base_ptr == nullptr) {
    return;
  }
  cudaSetDevice(device_index);
  ncclMemFree(base_ptr);
}

std::mutex allocations_mutex;
std::unordered_map<void*, std::shared_ptr<Allocation>> allocations;
std::unordered_map<uint64_t, std::shared_ptr<Allocation>> persistent_allocations;
std::unordered_map<uint64_t, bool> persistent_allocation_active;
std::atomic<size_t> configured_signal_pad_size{0};

size_t current_signal_pad_size() {
  const size_t configured = configured_signal_pad_size.load(std::memory_order_acquire);
  return configured == 0 ? kDefaultSignalPadSize : round_up(configured, kSignalPadAlignment);
}

void release_allocation(void* data_ptr, std::optional<uint64_t> alloc_id) {
  std::lock_guard<std::mutex> lock(allocations_mutex);
  if (alloc_id.has_value()) {
    persistent_allocation_active[*alloc_id] = false;
    return;
  }
  allocations.erase(data_ptr);
}

void release_persistent_allocation(uint64_t alloc_id) {
  std::lock_guard<std::mutex> lock(allocations_mutex);
  const auto allocation_it = persistent_allocations.find(alloc_id);
  TORCH_CHECK(
      allocation_it != persistent_allocations.end(),
      "no persistent symmetric allocation exists for alloc_id ",
      alloc_id);
  TORCH_CHECK(
      !persistent_allocation_active[alloc_id],
      "persistent symmetric allocation with alloc_id ",
      alloc_id,
      " is still active; release every tensor view before releasing its allocation");
  const auto data_ptr = allocation_it->second->data_ptr;
  allocations.erase(data_ptr);
  persistent_allocation_active.erase(alloc_id);
  persistent_allocations.erase(allocation_it);
}

struct AllocationRange {
  std::shared_ptr<Allocation> allocation;
  size_t offset;
  size_t size;
};

AllocationRange get_allocation_range(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), "tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0, "zero-sized tensors cannot be registered");

  const auto tensor_address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const auto tensor_size = static_cast<size_t>(tensor.numel()) * tensor.element_size();
  std::lock_guard<std::mutex> lock(allocations_mutex);
  for (const auto& [allocation_ptr, allocation] : allocations) {
    const auto allocation_begin = reinterpret_cast<uintptr_t>(allocation_ptr);
    const auto allocation_end = allocation_begin + allocation->user_size;
    TORCH_CHECK(allocation_end >= allocation_begin, "invalid symmetric allocation range");
    if (tensor_address >= allocation_begin && tensor_address <= allocation_end &&
        tensor_size <= static_cast<size_t>(allocation_end - tensor_address)) {
      TORCH_CHECK(
          tensor.get_device() == allocation->device_index,
          "tensor device does not match its symmetric allocation");
      return {allocation, static_cast<size_t>(tensor_address - allocation_begin), tensor_size};
    }
  }
  TORCH_CHECK(
      false,
      "tensor was not allocated by nccl_symm_mem.empty()/a tracked symmetric MemPool, "
      "or its view extends outside the original symmetric allocation");
}

std::shared_ptr<WindowState> get_or_create_window(
    const std::shared_ptr<Allocation>& allocation,
    int64_t comm_ptr,
    const std::string& group_key,
    int rank,
    int world_size) {
  const auto key = static_cast<uintptr_t>(comm_ptr);
  std::lock_guard<std::mutex> lock(allocation->mutex);
  const auto it = allocation->windows.find(key);
  if (it != allocation->windows.end()) {
    TORCH_CHECK(
        it->second->group_key_ == group_key && it->second->rank_ == rank &&
            it->second->world_size_ == world_size,
        "the same NCCL communicator pointer was used with incompatible group metadata");
    return it->second;
  }
  auto window = std::make_shared<WindowState>(allocation, comm_ptr, group_key, rank, world_size);
  allocation->windows.emplace(key, window);
  return window;
}

void release_window(
    const std::shared_ptr<Allocation>& allocation,
    const std::shared_ptr<WindowState>& window) {
  const auto key = reinterpret_cast<uintptr_t>(window->comm_);
  std::lock_guard<std::mutex> lock(allocation->mutex);
  const auto it = allocation->windows.find(key);
  if (it != allocation->windows.end() && it->second == window && window.use_count() == 2) {
    // The cache and this registration are the last owners. Erase the cache so
    // close() deregisters before ProcessGroupNCCL can destroy its communicator.
    allocation->windows.erase(it);
  }
}

class SymmetricRegistration final {
 public:
  SymmetricRegistration(
      const torch::Tensor& tensor,
      int64_t comm_ptr,
      std::string group_key,
      int rank,
      int world_size)
      : group_key_(std::move(group_key)) {
    const auto range = get_allocation_range(tensor);
    allocation_ = range.allocation;
    offset_ = range.offset;
    nbytes_ = static_cast<int64_t>(range.size);
    window_ = get_or_create_window(allocation_, comm_ptr, group_key_, rank, world_size);
  }

  void close() {
    if (closed_) {
      return;
    }
    closed_ = true;
    // A tracked MemPool allocation can outlive its ProcessGroup. Drop an idle
    // cached window now so its destructor deregisters while the communicator is
    // still valid. Keep it cached while another registration still uses it.
    release_window(allocation_, window_);
    window_.reset();
    allocation_.reset();
  }

  bool closed() const {
    return closed_;
  }

  int device_index() const {
    return allocation_->device_index;
  }

  int64_t nbytes() const {
    return nbytes_;
  }

  int64_t buffer_size() const {
    return static_cast<int64_t>(allocation_->user_size);
  }

  int64_t offset() const {
    return static_cast<int64_t>(offset_);
  }

  int64_t signal_pad_size() const {
    return static_cast<int64_t>(allocation_->signal_pad_size);
  }

  int rank() const {
    return window_->rank_;
  }

  int world_size() const {
    return window_->world_size_;
  }

  std::string group_key() const {
    return group_key_;
  }

  bool has_peer_access(int peer) {
    try {
      static_cast<void>(window_->peer_window_base(peer));
      return true;
    } catch (const c10::Error&) {
      return false;
    }
  }

  torch::Tensor get_buffer(
      int peer,
      const std::vector<int64_t>& sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(storage_offset >= 0, "storage_offset must be non-negative");
    const size_t byte_offset = offset_ +
        static_cast<size_t>(storage_offset) * c10::elementSize(dtype);
    const size_t byte_size = tensor_nbytes(sizes, dtype);
    TORCH_CHECK(
        byte_offset <= allocation_->user_size && byte_size <= allocation_->user_size - byte_offset,
        "requested peer buffer range exceeds the symmetric allocation");
    auto* base = static_cast<char*>(window_->peer_window_base(peer));
    // Peer mappings are GPU virtual addresses accessible from this process's
    // current device. from_blob must therefore use the local device, rather
    // than inferring a peer's physical device from the pointer.
    auto options = torch::TensorOptions().dtype(dtype).device(c10::Device(c10::kCUDA, device_index()));
    return at::for_blob(base + allocation_->signal_pad_size + byte_offset, sizes)
        .options(options)
        .target_device(c10::Device(c10::kCUDA, device_index()))
        .make_tensor();
  }

  torch::Tensor get_signal_pad(
      int peer,
      const std::vector<int64_t>& sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(allocation_->signal_pad_size != 0, "this allocation has no signal pad");
    TORCH_CHECK(storage_offset >= 0, "storage_offset must be non-negative");
    const auto actual_sizes = sizes.empty()
        ? std::vector<int64_t>{static_cast<int64_t>(allocation_->signal_pad_size / c10::elementSize(dtype))}
        : sizes;
    const size_t byte_offset = static_cast<size_t>(storage_offset) * c10::elementSize(dtype);
    const size_t byte_size = tensor_nbytes(actual_sizes, dtype);
    TORCH_CHECK(
        byte_offset <= allocation_->signal_pad_size &&
            byte_size <= allocation_->signal_pad_size - byte_offset,
        "requested signal pad range exceeds the configured signal pad size");
    auto* base = static_cast<char*>(window_->peer_window_base(peer));
    auto options = torch::TensorOptions().dtype(dtype).device(c10::Device(c10::kCUDA, device_index()));
    return at::for_blob(base + byte_offset, actual_sizes)
        .options(options)
        .target_device(c10::Device(c10::kCUDA, device_index()))
        .make_tensor();
  }

  int64_t multicast_ptr() {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    auto* ptr = static_cast<char*>(window_->multicast_data_ptr()) + offset_;
    return reinterpret_cast<int64_t>(ptr);
  }

  std::vector<int64_t> peer_buffer_ptrs() {
    std::vector<int64_t> result;
    result.reserve(static_cast<size_t>(world_size()));
    for (int peer = 0; peer < world_size(); ++peer) {
      try {
        auto* ptr = static_cast<char*>(window_->peer_window_base(peer));
        result.push_back(reinterpret_cast<int64_t>(ptr + allocation_->signal_pad_size + offset_));
      } catch (const c10::Error&) {
        result.push_back(0);
      }
    }
    return result;
  }

  void put_signal(int peer, int channel, int64_t /* timeout_ms */) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(peer >= 0 && peer < world_size(), "peer rank is out of range");
    TORCH_CHECK(channel >= 0, "channel must be non-negative");
    const auto fn = get_signal_fn();
    TORCH_CHECK(fn != nullptr, "one-sided signals require a runtime NCCL library exporting ncclSignal");
    c10::cuda::CUDAGuard device_guard(device_index());
    check_nccl(
        fn(peer, channel, 0, 0, window_->comm_, at::cuda::getCurrentCUDAStream()),
        "ncclSignal");
  }

  void wait_signal(int peer, int channel, int64_t /* timeout_ms */) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(peer >= 0 && peer < world_size(), "peer rank is out of range");
    TORCH_CHECK(channel >= 0, "channel must be non-negative");
    const auto fn = get_wait_signal_fn();
    TORCH_CHECK(fn != nullptr, "one-sided signals require a runtime NCCL library exporting ncclWaitSignal");
    NcclWaitSignalDesc descriptor{1, peer, channel, 0};
    c10::cuda::CUDAGuard device_guard(device_index());
    check_nccl(
        fn(1, &descriptor, window_->comm_, at::cuda::getCurrentCUDAStream()),
        "ncclWaitSignal");
  }

  void barrier(int channel, int64_t timeout_ms) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(channel >= 0, "channel must be non-negative");
    c10::cuda::CUDAGuard device_guard(device_index());
    for (int peer = 0; peer < world_size(); ++peer) {
      if (peer != rank()) {
        put_signal(peer, channel, timeout_ms);
      }
    }
    const auto wait_fn = get_wait_signal_fn();
    TORCH_CHECK(wait_fn != nullptr, "one-sided signals require a runtime NCCL library exporting ncclWaitSignal");
    std::vector<NcclWaitSignalDesc> descriptors;
    descriptors.reserve(static_cast<size_t>(world_size() - 1));
    for (int peer = 0; peer < world_size(); ++peer) {
      if (peer != rank()) {
        descriptors.push_back({1, peer, channel, 0});
      }
    }
    if (!descriptors.empty()) {
      check_nccl(
          wait_fn(
              static_cast<int>(descriptors.size()),
              descriptors.data(),
              window_->comm_,
              at::cuda::getCurrentCUDAStream()),
          "ncclWaitSignal(barrier)");
    }
  }

  py::tuple get_peer_cft_handle(int peer) {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    TORCH_CHECK(peer >= 0 && peer < world_size(), "peer rank is out of range");
    const auto fn = get_peer_cft_info_fn();
    TORCH_CHECK(fn != nullptr, "host-side CFT requires NCCL >= 2.31.2");
    uint32_t le_id = 0;
    size_t le_offset = 0;
    c10::cuda::CUDAGuard device_guard(device_index());
    check_nccl(
        fn(
            window_->window_,
            allocation_->signal_pad_size + offset_,
            peer,
            &le_id,
            &le_offset),
        "ncclGetPeerDeviceLeInfo");
    return py::make_tuple(le_id, le_offset);
  }

  py::tuple get_multimem_cft_handle() {
    TORCH_CHECK(!closed_, "symmetric registration is closed");
    const auto fn = get_multimem_cft_info_fn();
    TORCH_CHECK(fn != nullptr, "host-side CFT requires NCCL >= 2.31.2");
    uint32_t le_id = 0;
    size_t le_offset = 0;
    c10::cuda::CUDAGuard device_guard(device_index());
    check_nccl(
        fn(window_->window_, allocation_->signal_pad_size + offset_, &le_id, &le_offset),
        "ncclGetMultimemDeviceLeInfo");
    return py::make_tuple(le_id, le_offset);
  }

 private:
  static size_t tensor_nbytes(const std::vector<int64_t>& sizes, c10::ScalarType dtype) {
    size_t numel = 1;
    for (const auto size : sizes) {
      TORCH_CHECK(size >= 0, "sizes must be non-negative");
      TORCH_CHECK(
          size == 0 || numel <= SIZE_MAX / static_cast<size_t>(size),
          "tensor size overflow");
      numel *= static_cast<size_t>(size);
    }
    TORCH_CHECK(numel <= SIZE_MAX / c10::elementSize(dtype), "tensor byte size overflow");
    return numel * c10::elementSize(dtype);
  }

  std::shared_ptr<Allocation> allocation_;
  std::shared_ptr<WindowState> window_;
  std::string group_key_;
  size_t offset_{0};
  int64_t nbytes_{0};
  bool closed_{false};
};

std::shared_ptr<Allocation> make_owned_allocation(
    int device_index,
    size_t nbytes,
    std::optional<uint64_t> alloc_id) {
  std::lock_guard<std::mutex> lock(allocations_mutex);
  if (alloc_id.has_value()) {
    const auto persistent_it = persistent_allocations.find(*alloc_id);
    if (persistent_it != persistent_allocations.end()) {
      TORCH_CHECK(
          !persistent_allocation_active[*alloc_id],
          "a persistent symmetric allocation with alloc_id ",
          *alloc_id,
          " is still active");
      const auto& allocation = persistent_it->second;
      TORCH_CHECK(
          allocation->device_index == device_index && allocation->user_size == nbytes,
          "persistent symmetric allocation alloc_id ",
          *alloc_id,
          " was requested with a different device or size");
      persistent_allocation_active[*alloc_id] = true;
      allocations[allocation->data_ptr] = allocation;
      return allocation;
    }
  }
  auto allocation = std::make_shared<Allocation>(device_index, nbytes, current_signal_pad_size());
  allocations.emplace(allocation->data_ptr, allocation);
  if (alloc_id.has_value()) {
    persistent_allocations.emplace(*alloc_id, allocation);
    persistent_allocation_active[*alloc_id] = true;
  }
  return allocation;
}

torch::Tensor empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device,
    std::optional<uint64_t> alloc_id) {
  TORCH_CHECK(device.is_cuda(), "device must be CUDA");
  size_t numel = 1;
  for (const auto size : sizes) {
    TORCH_CHECK(size >= 0, "sizes must be non-negative");
    TORCH_CHECK(
        size == 0 || numel <= SIZE_MAX / static_cast<size_t>(size),
        "symmetric allocation size overflow");
    numel *= static_cast<size_t>(size);
  }
  TORCH_CHECK(numel <= SIZE_MAX / c10::elementSize(dtype), "symmetric allocation byte size overflow");
  const size_t nbytes = numel * c10::elementSize(dtype);
  TORCH_CHECK(nbytes > 0, "zero-sized symmetric allocations are not supported");

  auto allocation = make_owned_allocation(device.index(), nbytes, alloc_id);
  const auto ptr = allocation->data_ptr;
  const auto options = torch::TensorOptions().dtype(dtype).device(device);
  return torch::from_blob(
      ptr,
      sizes,
      [ptr, alloc_id](void*) { release_allocation(ptr, alloc_id); },
      options);
}

void track_tensor(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), "tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), "tracked symmetric MemPool tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0, "zero-sized tensors cannot be tracked");
  const auto address = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
  CUdeviceptr base = 0;
  size_t size = 0;
  const auto range_result = cuMemGetAddressRange(&base, &size, address);
  TORCH_CHECK(
      range_result == CUDA_SUCCESS && base != 0 && size > 0,
      "tensor is not a CUDA VMM allocation and cannot be tracked as symmetric memory");
  Allocation::validate_vmm(reinterpret_cast<void*>(base));
  const auto tensor_address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const auto base_address = static_cast<uintptr_t>(base);
  TORCH_CHECK(tensor_address >= base_address, "invalid CUDA VMM address range");
  const auto offset = static_cast<size_t>(tensor_address - base_address);
  const auto tensor_size = static_cast<size_t>(tensor.numel()) * tensor.element_size();
  TORCH_CHECK(offset <= size && tensor_size <= size - offset, "tensor extends beyond its CUDA VMM range");

  std::lock_guard<std::mutex> lock(allocations_mutex);
  const auto data_ptr = reinterpret_cast<void*>(base);
  if (allocations.find(data_ptr) == allocations.end()) {
    allocations.emplace(
        data_ptr,
        std::make_shared<Allocation>(tensor.get_device(), data_ptr, size, size));
  }
}

bool is_symmetric_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_cuda() || !tensor.is_contiguous() || tensor.numel() <= 0) {
    return false;
  }
  try {
    static_cast<void>(get_allocation_range(tensor));
    return true;
  } catch (const c10::Error&) {
    return false;
  }
}

void set_signal_pad_size(size_t size) {
  TORCH_CHECK(size > 0, "signal pad size must be positive");
  std::lock_guard<std::mutex> lock(allocations_mutex);
  TORCH_CHECK(
      allocations.empty() && persistent_allocations.empty(),
      "signal pad size must be configured before creating symmetric allocations");
  configured_signal_pad_size.store(size, std::memory_order_release);
}

py::dict capabilities() {
  const int version = runtime_nccl_version();
  py::dict result;
  result["windows"] = version >= NCCL_VERSION(2, 27, 0);
  result["device_symmetric_memory"] = version >= NCCL_VERSION(2, 28, 4);
  result["one_sided_signals"] = version >= NCCL_VERSION(2, 29, 0) &&
      get_signal_fn() != nullptr && get_wait_signal_fn() != nullptr;
  result["peer_buffers"] = get_peer_device_pointer_fn() != nullptr;
  result["multicast"] = get_multimem_device_pointer_fn() != nullptr;

  // This is the header/toolkit gate equivalent to NCCL's #if CUDA_VERSION
  // >= 13030 CFT build gate, plus the NCCL 2.31.2 host-query API introduction.
  result["host_cft_build"] = static_cast<bool>(NCCL_SYMM_MEM_HAS_HOST_CFT_BUILD);
  result["nccl_header_version"] = NCCL_VERSION_CODE;
  result["cuda_header_version"] = CUDA_VERSION;

  // A positive result means the extension's build is CFT-capable and the
  // loaded NCCL runtime exports the 2.31.2+ query API. It deliberately does
  // not claim that a specific communicator is CFT-ready: that can only be
  // established after rendezvous through get_*_cft_handle().
  result["host_cft"] = NCCL_SYMM_MEM_HAS_HOST_CFT_BUILD &&
      version >= NCCL_VERSION(2, 31, 2) &&
      get_peer_cft_info_fn() != nullptr &&
      get_multimem_cft_info_fn() != nullptr;
  return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NCCL symmetric-memory allocation and window extension";
  m.def(
      "empty",
      &empty,
      py::arg("sizes"),
      py::arg("dtype"),
      py::arg("device"),
      py::arg("alloc_id") = py::none());
  m.def("track_tensor", &track_tensor, py::arg("tensor"));
  m.def("release_persistent_allocation", &release_persistent_allocation, py::arg("alloc_id"));
  m.def("is_symmetric_tensor", &is_symmetric_tensor, py::arg("tensor"));
  m.def("nccl_version", &runtime_nccl_version);
  m.def("capabilities", &capabilities);
  m.def("get_signal_pad_size", &current_signal_pad_size);
  m.def("set_signal_pad_size", &set_signal_pad_size, py::arg("size"));
  m.def("supports_symmetric_windows", []() {
    return runtime_nccl_version() >= NCCL_VERSION(2, 27, 0);
  });
  py::class_<SymmetricRegistration, std::shared_ptr<SymmetricRegistration>>(
      m, "_SymmetricRegistration")
      .def(py::init<const torch::Tensor&, int64_t, std::string, int, int>())
      .def("close", &SymmetricRegistration::close)
      .def("get_buffer", &SymmetricRegistration::get_buffer, py::arg("peer"), py::arg("sizes"), py::arg("dtype"), py::arg("storage_offset") = 0)
      .def("get_signal_pad", &SymmetricRegistration::get_signal_pad, py::arg("peer"), py::arg("sizes") = std::vector<int64_t>{}, py::arg("dtype") = c10::kUInt32, py::arg("storage_offset") = 0)
      .def("has_peer_access", &SymmetricRegistration::has_peer_access, py::arg("peer"))
      .def("barrier", &SymmetricRegistration::barrier, py::arg("channel") = 0, py::arg("timeout_ms") = 0)
      .def("put_signal", &SymmetricRegistration::put_signal, py::arg("peer"), py::arg("channel") = 0, py::arg("timeout_ms") = 0)
      .def("wait_signal", &SymmetricRegistration::wait_signal, py::arg("peer"), py::arg("channel") = 0, py::arg("timeout_ms") = 0)
      .def("get_peer_cft_handle", &SymmetricRegistration::get_peer_cft_handle, py::arg("peer"))
      .def("get_multimem_cft_handle", &SymmetricRegistration::get_multimem_cft_handle)
      .def_property_readonly("closed", &SymmetricRegistration::closed)
      .def_property_readonly("device_index", &SymmetricRegistration::device_index)
      .def_property_readonly("nbytes", &SymmetricRegistration::nbytes)
      .def_property_readonly("buffer_size", &SymmetricRegistration::buffer_size)
      .def_property_readonly("offset", &SymmetricRegistration::offset)
      .def_property_readonly("signal_pad_size", &SymmetricRegistration::signal_pad_size)
      .def_property_readonly("rank", &SymmetricRegistration::rank)
      .def_property_readonly("world_size", &SymmetricRegistration::world_size)
      .def_property_readonly("group_key", &SymmetricRegistration::group_key)
      .def_property_readonly("multicast_ptr", &SymmetricRegistration::multicast_ptr)
      .def_property_readonly("peer_buffer_ptrs", &SymmetricRegistration::peer_buffer_ptrs);
}
