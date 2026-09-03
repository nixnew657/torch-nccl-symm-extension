#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <nccl.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

void check_nccl(ncclResult_t result, const char* operation) {
  TORCH_CHECK(
      result == ncclSuccess,
      operation,
      " failed: ",
      ncclGetErrorString(result));
}

struct Allocation {
  Allocation(int device_index, size_t size) : device_index(device_index), size(size) {
    c10::cuda::CUDAGuard device_guard(device_index);
    check_nccl(ncclMemAlloc(&ptr, size), "ncclMemAlloc");
  }

  ~Allocation() {
    if (ptr == nullptr) {
      return;
    }
    // Destructors can run during interpreter teardown. There is no useful
    // recovery path here, and ncclMemFree does not report through exceptions.
    cudaSetDevice(device_index);
    ncclMemFree(ptr);
  }

  int device_index;
  size_t size;
  void* ptr{nullptr};
};

std::mutex allocations_mutex;
std::unordered_map<void*, std::shared_ptr<Allocation>> allocations;

void release_allocation(void* ptr) {
  std::lock_guard<std::mutex> lock(allocations_mutex);
  allocations.erase(ptr);
}

std::shared_ptr<Allocation> get_allocation(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), "tensor must be a CUDA tensor");
  TORCH_CHECK(tensor.storage_offset() == 0, "only the original tensor can be registered");
  const auto ptr = tensor.data_ptr();
  std::lock_guard<std::mutex> lock(allocations_mutex);
  const auto it = allocations.find(ptr);
  TORCH_CHECK(
      it != allocations.end(),
      "tensor was not allocated by nccl_symm_mem.empty(); only the original "
      "tensor (not a view) can be registered");
  TORCH_CHECK(
      tensor.numel() * tensor.element_size() == static_cast<int64_t>(it->second->size),
      "only a full allocation can be registered as an NCCL symmetric window");
  return it->second;
}

class SymmetricRegistration final {
 public:
  SymmetricRegistration(
      const torch::Tensor& tensor,
      int64_t comm_ptr,
      std::string group_key)
      : allocation_(get_allocation(tensor)),
        comm_(reinterpret_cast<ncclComm_t>(comm_ptr)),
        group_key_(std::move(group_key)),
        device_index_(allocation_->device_index),
        nbytes_(static_cast<int64_t>(allocation_->size)) {
    TORCH_CHECK(comm_ != nullptr, "NCCL communicator is not initialized");
    TORCH_CHECK(!group_key_.empty(), "group_key must not be empty");
    c10::cuda::CUDAGuard device_guard(device_index_);
    check_nccl(
        ncclCommWindowRegister(
            comm_, allocation_->ptr, allocation_->size, &window_, NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)");
  }

  ~SymmetricRegistration() {
    close_noexcept();
  }

  void close() {
    TORCH_CHECK(!closed_, "symmetric NCCL registration is already closed");
    c10::cuda::CUDAGuard device_guard(device_index_);
    check_nccl(ncclCommWindowDeregister(comm_, window_), "ncclCommWindowDeregister");
    window_ = nullptr;
    closed_ = true;
    allocation_.reset();
  }

  bool closed() const {
    return closed_;
  }

  int device_index() const {
    return device_index_;
  }

  int64_t nbytes() const {
    return nbytes_;
  }

  std::string group_key() const {
    return group_key_;
  }

 private:
  void close_noexcept() noexcept {
    if (closed_ || window_ == nullptr || comm_ == nullptr) {
      return;
    }
    cudaSetDevice(device_index_);
    ncclCommWindowDeregister(comm_, window_);
    window_ = nullptr;
    closed_ = true;
    allocation_.reset();
  }

  std::shared_ptr<Allocation> allocation_;
  ncclComm_t comm_;
  std::string group_key_;
  int device_index_;
  int64_t nbytes_;
  ncclWindow_t window_{nullptr};
  bool closed_{false};
};

torch::Tensor empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device) {
  TORCH_CHECK(device.is_cuda(), "device must be CUDA");
  TORCH_CHECK(!sizes.empty(), "scalar tensors are not supported by this extension");

  size_t numel = 1;
  for (const auto size : sizes) {
    TORCH_CHECK(size >= 0, "sizes must be non-negative");
    numel *= static_cast<size_t>(size);
  }
  const size_t nbytes = numel * c10::elementSize(dtype);
  TORCH_CHECK(nbytes > 0, "zero-sized symmetric allocations are not supported");

  auto allocation = std::make_shared<Allocation>(device.index(), nbytes);
  const auto ptr = allocation->ptr;
  {
    std::lock_guard<std::mutex> lock(allocations_mutex);
    allocations.emplace(ptr, allocation);
  }

  const auto options = torch::TensorOptions().dtype(dtype).device(device);
  return torch::from_blob(
      ptr,
      sizes,
      [ptr](void*) { release_allocation(ptr); },
      options);
}

bool is_symmetric_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_cuda() || tensor.storage_offset() != 0) {
    return false;
  }
  const auto ptr = tensor.data_ptr();
  std::lock_guard<std::mutex> lock(allocations_mutex);
  const auto it = allocations.find(ptr);
  return it != allocations.end() &&
      tensor.numel() * tensor.element_size() == static_cast<int64_t>(it->second->size);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NCCL symmetric-window allocation extension";
  m.def("empty", &empty, py::arg("sizes"), py::arg("dtype"), py::arg("device"));
  m.def("is_symmetric_tensor", &is_symmetric_tensor, py::arg("tensor"));
  m.def("nccl_version", []() {
    int version = 0;
    check_nccl(ncclGetVersion(&version), "ncclGetVersion");
    return version;
  });
  m.def("supports_symmetric_windows", []() {
    return NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0);
  });
  py::class_<SymmetricRegistration, std::shared_ptr<SymmetricRegistration>>(
      m, "_SymmetricRegistration")
      .def(py::init<const torch::Tensor&, int64_t, std::string>())
      .def("close", &SymmetricRegistration::close)
      .def_property_readonly("closed", &SymmetricRegistration::closed)
      .def_property_readonly("device_index", &SymmetricRegistration::device_index)
      .def_property_readonly("nbytes", &SymmetricRegistration::nbytes)
      .def_property_readonly("group_key", &SymmetricRegistration::group_key);
}
