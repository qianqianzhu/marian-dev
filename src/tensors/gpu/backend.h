/* All or part of this file was contributed by NVIDIA under license:
 *   Copyright (C) 2020 NVIDIA Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "common/config.h"
#include "tensors/backend.h"  // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"
#include "common/logging.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

namespace marian {
namespace gpu {

// @TODO: in the future this should pobably become a fully fledged CudaInfo class with many attributes
struct CudaCompute {
  int major;
  int minor;
};

class Backend : public marian::Backend {
private:
  void setCudaComputeCapability() {
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.major, cudaDevAttrComputeCapabilityMajor, (int)deviceId_.no));
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.minor, cudaDevAttrComputeCapabilityMinor, (int)deviceId_.no));
  }

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setCudaComputeCapability();
  }

  ~Backend() {
    setDevice();
    if(cusparseHandle_) {
      cusparseDestroy(cusparseHandle_);
      cusparseHandle_ = 0;
    }
    if(cublasHandle_) {
      cublasDestroy(cublasHandle_);
      cublasHandle_ = 0;
    }
  }

  void setDevice() override { CUDA_CHECK(cudaSetDevice((int)deviceId_.no)); }

  void synchronize() override { CUDA_CHECK(cudaStreamSynchronize(0)); }

  cublasHandle_t getCublasHandle() {
    if(!cublasHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cublasCreate(&cublasHandle_);
      cublasSetStream(cublasHandle_, cudaStreamPerThread);
    }
    return cublasHandle_;
  }

  cusparseHandle_t getCusparseHandle() {
    if(!cusparseHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cusparseCreate(&cusparseHandle_);
      cusparseSetStream(cusparseHandle_, cudaStreamPerThread);  
    }
    return cusparseHandle_;
  }

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  void setOptimized(bool optimize) override {
    LOG_ONCE(info, "setOptimized() not supported for GPU_{}", optimize);
  }
  bool isOptimized() override {
    LOG_ONCE(info, "isOptimized() not supported for GPU");
    return false;
  };

  // for CPU and GPU selects different typesf
  void setGemmType(std::string gemmType) override {
      if (gemmType == "float32")     gemmType_ = GemmType::Float32;
      else if (gemmType.find("int8tensorcore") == 0) gemmType_ = GemmType::int8tensorcore;
      else ABORT("Unknown GEMM type - '{}'", gemmType);
  }
  GemmType getGemmType() override { return gemmType_; }

  // for CPU, sets quantization range of weight matrices for the inference.
  // for GPU, there's no quantization. so, it does nothing.
  void setQuantizeRange(float range) override {
    LOG_ONCE(info, "setQuantizeRange() not supported for GPU_{}", range);
  }
  float getQuantizeRange() override {
    LOG_ONCE(info, "getQuantizeRange() not supported for GPU");
    return 0.f;
  }

  CudaCompute getCudaComputeCapability() { return compute_; }

private:
  cublasHandle_t cublasHandle_{0};     // make sure it's 0, so it can be initalized lazily
  cusparseHandle_t cusparseHandle_{0}; // as above
  CudaCompute compute_;
};
}  // namespace gpu
}  // namespace marian
