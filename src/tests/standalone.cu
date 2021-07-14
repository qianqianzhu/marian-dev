#include <iostream>
#include <sstream>
#include <vector>

#include "tensors/gpu/cuda_helpers.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"

typedef int8_t abtype;

/**************************CUTLASS code begins here***********************/
inline const char *cutlassGetErrorString(cutlass::Status &status) {
  switch(status) {
    case cutlass::Status::kSuccess: return "Operation was successful.";
    case cutlass::Status::kErrorMisalignedOperand: return "Operands fail alignment requirements.";
    case cutlass::Status::kErrorInvalidLayout: return "Layout fails alignment requirement.";
    case cutlass::Status::kErrorInvalidProblem:
      return "Specified problem size is not supported by operator.";
    case cutlass::Status::kErrorNotSupported:
      return "Operation is not supported on current device.";
    case cutlass::Status::kErrorWorkspaceNull:
      return "The given workspace is null when it is required to be non-null";
    case cutlass::Status::kErrorInternal: return "An error within CUTLASS occurred.";
    case cutlass::Status::kInvalid: return "Status is unspecified.";
    case cutlass::Status::kErrorInvalidDataType: break;
    case cutlass::Status::kErrorArchMismatch: break;
    case cutlass::Status::kErrorInsufficientDriver: break;
  }
  return "Unknown CUTLASS status. Update this section of the code.";
}

#define CUTLASS_CHECK(expr)                   \
  do {                                        \
    cutlass::Status rc = (expr);              \
    ABORT_IF(rc != cutlass::Status::kSuccess, \
             "Cutlass Error: {} - {}:{}: {}", \
             cutlassGetErrorString(rc),       \
             __FILE__,                        \
             __LINE__,                        \
             #expr);                          \
  } while(0)

/*Cutlass matrices*/
using ElementOutput = float;
using ElementAccumulator = int32_t;
using ElementCompute = float;
/*TensorOp matrices*/

#ifdef CUTLASS_SM75
// Compute arch
using SmArch = cutlass::arch::Sm75;
// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock
    = cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16
// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??
// Number of pipelines you want to use
constexpr int NumStages = 2;
#else
// Compute arch
using SmArch = cutlass::arch::Sm80;
// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock
    = cutlass::gemm::GemmShape<128, 128, 64>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 8
// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??
// Number of pipelines you want to use
constexpr int NumStages = 3;
#endif

#if 0  // These settings are for A100 which have bigger shared memory. Otherwise they crash with the
       // 3090tis that we have
// Compute arch
    using SmArch = cutlass::arch::Sm80;
    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 128>;  // <- threadblock tile M = 128, N = 128, K = 16
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 16
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 8
    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??
    // Number of pipelines you want to use
    constexpr int NumStages = 4;
#endif

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
    // memory access. For a byte, it's 16
    // elements. This becomes the vector width of
    // math instructions in the epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementCompute>;     // <- data type for alpha/beta in linear combination function

using EpilogueOpRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
    // memory access. For a byte, it's 16
    // elements. This becomes the vector width of
    // math instructions in the epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementCompute>;     // <- data type for alpha/beta in linear combination function

using CutlassGemmTensorOp
    = cutlass::gemm::device::Gemm<int8_t,                          // ElementA
                                  cutlass::layout::RowMajor,       // LayoutA
                                  int8_t,                          // ElementB
                                  cutlass::layout::ColumnMajor,    // LayoutB
                                  float,                           // ElementOutput
                                  cutlass::layout::ColumnMajor,    // LayoutOutput
                                  int32_t,                         // ElementAccumulator
                                  cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
                                  SmArch,  // tag indicating target GPU compute architecture //@TODO
                                           // this should change, probably
                                  ShapeMMAThreadBlock,
                                  ShapeMMAWarp,
                                  ShapeMMAOp,
                                  EpilogueOp,
                                  SwizzleThreadBlock,
                                  NumStages>;
using CutlassGemmTensorOpRelu
    = cutlass::gemm::device::Gemm<int8_t,                          // ElementA
                                  cutlass::layout::RowMajor,       // LayoutA
                                  int8_t,                          // ElementB
                                  cutlass::layout::ColumnMajor,    // LayoutB
                                  float,                           // ElementOutput
                                  cutlass::layout::ColumnMajor,    // LayoutOutput
                                  int32_t,                         // ElementAccumulator
                                  cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
                                  SmArch,  // tag indicating target GPU compute architecture //@TODO
                                           // this should change, probably
                                  ShapeMMAThreadBlock,
                                  ShapeMMAWarp,
                                  ShapeMMAOp,
                                  EpilogueOpRelu,
                                  SwizzleThreadBlock,
                                  NumStages>;
/*Non TensorOp matrices*/
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                              1, /*@TODO should be something
                                                                    different? like 32/64/128?*/
                                                              ElementAccumulator,
                                                              ElementCompute>;

using EpilogueRelu
    = cutlass::epilogue::thread::LinearCombinationRelu<ElementOutput,
                                                       1, /*@TODO should be something different?
                                                             like 32/64/128?*/
                                                       ElementAccumulator,
                                                       ElementCompute>;

using ColumnMajor = cutlass::layout::ColumnMajor;
using ColumnMajorT
    = cutlass::layout::RowMajor;  // Transposing in cutlass is done by changing the input from
                                  // RowMajor to ColumnMajor. Care of the output
// using RowMajor = cutlass::layout::RowMajor;
using CutlassGemmTT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                  ColumnMajorT,  // Layout of A matrix
                                                  int8_t,        // Data-type of B matrix
                                                  ColumnMajorT,  // Layout of B matrix
                                                  float,         // Data-type of C matrix
                                                  ColumnMajor,   // Layout of C matrix
                                                  int32_t,       // Accumulator
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm75,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  Epilogue>;
using CutlassGemmNT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                  ColumnMajor,   // Layout of A matrix
                                                  int8_t,        // Data-type of B matrix
                                                  ColumnMajorT,  // Layout of B matrix
                                                  float,         // Data-type of C matrix
                                                  ColumnMajor,   // Layout of C matrix
                                                  int32_t,       // Accumulator
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm75,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  Epilogue>;

using CutlassGemmTN = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                  ColumnMajorT,  // Layout of A matrix
                                                  int8_t,        // Data-type of B matrix
                                                  ColumnMajor,   // Layout of B matrix
                                                  float,         // Data-type of C matrix
                                                  ColumnMajor,   // Layout of C matrix
                                                  int32_t,       // Accumulator
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm75,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  Epilogue>;

using CutlassGemmNN = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  int8_t,       // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor,  // Layout of C matrix
                                                  int32_t,      // Accumulator
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm75,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  Epilogue>;

using CutlassGemmTTRelu = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                      ColumnMajorT,  // Layout of A matrix
                                                      int8_t,        // Data-type of B matrix
                                                      ColumnMajorT,  // Layout of B matrix
                                                      float,         // Data-type of C matrix
                                                      ColumnMajor,   // Layout of C matrix
                                                      int32_t,       // Accumulator
                                                      cutlass::arch::OpClassSimt,
                                                      cutlass::arch::Sm75,
                                                      ThreadBlockShape,
                                                      WarpShape,
                                                      InstructionShape,
                                                      EpilogueRelu>;
using CutlassGemmNTRelu = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                      ColumnMajor,   // Layout of A matrix
                                                      int8_t,        // Data-type of B matrix
                                                      ColumnMajorT,  // Layout of B matrix
                                                      float,         // Data-type of C matrix
                                                      ColumnMajor,   // Layout of C matrix
                                                      int32_t,       // Accumulator
                                                      cutlass::arch::OpClassSimt,
                                                      cutlass::arch::Sm75,
                                                      ThreadBlockShape,
                                                      WarpShape,
                                                      InstructionShape,
                                                      EpilogueRelu>;

using CutlassGemmTNRelu = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                      ColumnMajorT,  // Layout of A matrix
                                                      int8_t,        // Data-type of B matrix
                                                      ColumnMajor,   // Layout of B matrix
                                                      float,         // Data-type of C matrix
                                                      ColumnMajor,   // Layout of C matrix
                                                      int32_t,       // Accumulator
                                                      cutlass::arch::OpClassSimt,
                                                      cutlass::arch::Sm75,
                                                      ThreadBlockShape,
                                                      WarpShape,
                                                      InstructionShape,
                                                      EpilogueRelu>;

using CutlassGemmNNRelu = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                      ColumnMajor,  // Layout of A matrix
                                                      int8_t,       // Data-type of B matrix
                                                      ColumnMajor,  // Layout of B matrix
                                                      float,        // Data-type of C matrix
                                                      ColumnMajor,  // Layout of C matrix
                                                      int32_t,      // Accumulator
                                                      cutlass::arch::OpClassSimt,
                                                      cutlass::arch::Sm75,
                                                      ThreadBlockShape,
                                                      WarpShape,
                                                      InstructionShape,
                                                      EpilogueRelu>;

/*Non-Epilogue functions, as they are faster (for now)*/
using CutlassGemmTensorOpunfused
    = cutlass::gemm::device::Gemm<int8_t,                          // ElementA
                                  cutlass::layout::RowMajor,       // LayoutA
                                  int8_t,                          // ElementB
                                  cutlass::layout::ColumnMajor,    // LayoutB
                                  int32_t,                         // ElementOutput
                                  cutlass::layout::ColumnMajor,    // LayoutOutput
                                  int32_t,                         // ElementAccumulator
                                  cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
                                  cutlass::arch::Sm75>;

using CutlassGemmTTunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                         ColumnMajorT,  // Layout of A matrix
                                                         int8_t,        // Data-type of B matrix
                                                         ColumnMajorT,  // Layout of B matrix
                                                         int32_t,       // Data-type of C matrix
                                                         ColumnMajor,   // Layout of C matrix
                                                         int32_t>;      // Accumulator

using CutlassGemmNTunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                         ColumnMajor,   // Layout of A matrix
                                                         int8_t,        // Data-type of B matrix
                                                         ColumnMajorT,  // Layout of B matrix
                                                         int32_t,       // Data-type of C matrix
                                                         ColumnMajor,   // Layout of C matrix
                                                         int32_t>;      // Accumulator

using CutlassGemmTNunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                         ColumnMajorT,  // Layout of A matrix
                                                         int8_t,        // Data-type of B matrix
                                                         ColumnMajor,   // Layout of B matrix
                                                         int32_t,       // Data-type of C matrix
                                                         ColumnMajor,   // Layout of C matrix
                                                         int32_t>;      // Accumulator

using CutlassGemmNNunfused = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                         ColumnMajor,  // Layout of A matrix
                                                         int8_t,       // Data-type of B matrix
                                                         ColumnMajor,  // Layout of B matrix
                                                         int32_t,      // Data-type of C matrix
                                                         ColumnMajor,  // Layout of C matrix
                                                         int32_t>;     // Accumulator

cutlass::Status cutlass_igemm_nn(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    float *alpha,
    int8_t const *A,
    int lda,
    int8_t const *B,
    int ldb,
    float *beta,
    float *C,
    int ldc,
    bool tensorCore, /*We want this to be true for best performance*/
    bool fused, /* fused unquantisation (and bias addition (and activation function) if those are
                   present). Should be true for best performance */
    float *bias,
    bool doRelu) {
  // printf("Success M:%d N:%d K%d, Relu:%d, bias:%d\n", M, N, K, (int)doRelu,
  // (int)(bias!=nullptr));
  float *Csrc;
  int ldcSRC;
  if(bias) { /* This is only available for the fused option. Beta needs to be 1? */
    Csrc = bias;
    ldcSRC = 0; /*Having a stride of 0 enables bias broadcast*/
  } else {
    Csrc = C;
    ldcSRC = ldc;
  }
  if(fused) {
    if(doRelu) {
      if(tensorCore) {
        CutlassGemmTensorOpRelu gemm_operator;
        CutlassGemmTensorOpRelu::Arguments args(
            {M, N, K},       // Gemm Problem dimensions
            {A, lda},        // Tensor-ref for source matrix A
            {B, ldb},        // Tensor-ref for source matrix B
            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
            {C, ldc},  // Tensor-ref for destination matrix D (may be different memory than source C
                       // matrix)
            {alpha, beta});  // Scalars used in the Epilogue
        return gemm_operator(args);
      } else {
        if(!transA && !transB) {
          CutlassGemmNNRelu gemm_operator;
          CutlassGemmNNRelu::Arguments args({M, N, K},       // Gemm Problem dimensions
                                            {A, lda},        // Tensor-ref for source matrix A
                                            {B, ldb},        // Tensor-ref for source matrix B
                                            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                            {C, ldc},  // Tensor-ref for destination matrix D (may
                                                       // be different memory than source C matrix)
                                            {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else if(transA && !transB) {
          CutlassGemmTNRelu gemm_operator;
          CutlassGemmTNRelu::Arguments args({M, N, K},       // Gemm Problem dimensions
                                            {A, lda},        // Tensor-ref for source matrix A
                                            {B, ldb},        // Tensor-ref for source matrix B
                                            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                            {C, ldc},  // Tensor-ref for destination matrix D (may
                                                       // be different memory than source C matrix)
                                            {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else if(!transA && transB) {
          CutlassGemmNTRelu gemm_operator;
          CutlassGemmNTRelu::Arguments args({M, N, K},       // Gemm Problem dimensions
                                            {A, lda},        // Tensor-ref for source matrix A
                                            {B, ldb},        // Tensor-ref for source matrix B
                                            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                            {C, ldc},  // Tensor-ref for destination matrix D (may
                                                       // be different memory than source C matrix)
                                            {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else {  // Final case (transA && transB)
          CutlassGemmTTRelu gemm_operator;
          CutlassGemmTTRelu::Arguments args({M, N, K},       // Gemm Problem dimensions
                                            {A, lda},        // Tensor-ref for source matrix A
                                            {B, ldb},        // Tensor-ref for source matrix B
                                            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                            {C, ldc},  // Tensor-ref for destination matrix D (may
                                                       // be different memory than source C matrix)
                                            {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        }
      }
    } else {
      if(tensorCore) {
        CutlassGemmTensorOp gemm_operator;
        CutlassGemmTensorOp::Arguments args({M, N, K},       // Gemm Problem dimensions
                                            {A, lda},        // Tensor-ref for source matrix A
                                            {B, ldb},        // Tensor-ref for source matrix B
                                            {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                            {C, ldc},  // Tensor-ref for destination matrix D (may
                                                       // be different memory than source C matrix)
                                            {alpha, beta});  // Scalars used in the Epilogue
        return gemm_operator(args);
      } else {
        if(!transA && !transB) {
          CutlassGemmNN gemm_operator;
          CutlassGemmNN::Arguments args({M, N, K},       // Gemm Problem dimensions
                                        {A, lda},        // Tensor-ref for source matrix A
                                        {B, ldb},        // Tensor-ref for source matrix B
                                        {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                        {C, ldc},  // Tensor-ref for destination matrix D (may be
                                                   // different memory than source C matrix)
                                        {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else if(transA && !transB) {
          CutlassGemmTN gemm_operator;
          CutlassGemmTN::Arguments args({M, N, K},       // Gemm Problem dimensions
                                        {A, lda},        // Tensor-ref for source matrix A
                                        {B, ldb},        // Tensor-ref for source matrix B
                                        {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                        {C, ldc},  // Tensor-ref for destination matrix D (may be
                                                   // different memory than source C matrix)
                                        {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else if(!transA && transB) {
          CutlassGemmNT gemm_operator;
          CutlassGemmNT::Arguments args({M, N, K},       // Gemm Problem dimensions
                                        {A, lda},        // Tensor-ref for source matrix A
                                        {B, ldb},        // Tensor-ref for source matrix B
                                        {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                        {C, ldc},  // Tensor-ref for destination matrix D (may be
                                                   // different memory than source C matrix)
                                        {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        } else {  // Final case (transA && transB)
          CutlassGemmTT gemm_operator;
          CutlassGemmTT::Arguments args({M, N, K},       // Gemm Problem dimensions
                                        {A, lda},        // Tensor-ref for source matrix A
                                        {B, ldb},        // Tensor-ref for source matrix B
                                        {Csrc, ldcSRC},  // Tensor-ref for source matrix C
                                        {C, ldc},  // Tensor-ref for destination matrix D (may be
                                                   // different memory than source C matrix)
                                        {alpha, beta});  // Scalars used in the Epilogue
          return gemm_operator(args);
        }
      }
    }
  } else {
    static const int32_t constexpr alpha_int = 1;
    static const int32_t constexpr beta_int = 0;
    if(tensorCore) {
      CutlassGemmTensorOpunfused gemm_operator;
      CutlassGemmTensorOpunfused::Arguments args(
          {M, N, K},            // Gemm Problem dimensions
          {A, lda},             // Tensor-ref for source matrix A
          {B, ldb},             // Tensor-ref for source matrix B
          {(int32_t *)C, ldc},  // Tensor-ref for source matrix C
          {(int32_t *)C, ldc},  // Tensor-ref for destination matrix D (may be different memory than
                                // source C matrix)
          {alpha_int, beta_int});  // Scalars used in the Epilogue
      return gemm_operator(args);
    } else {
      if(!transA && !transB) {
        CutlassGemmNNunfused gemm_operator;
        CutlassGemmNNunfused::Arguments args(
            {M, N, K},               // Gemm Problem dimensions
            {A, lda},                // Tensor-ref for source matrix A
            {B, ldb},                // Tensor-ref for source matrix B
            {(int32_t *)C, ldc},     // Tensor-ref for source matrix C
            {(int32_t *)C, ldc},     // Tensor-ref for destination matrix D (may be different memory
                                     // than source C matrix)
            {alpha_int, beta_int});  // Scalars used in the Epilogue
        return gemm_operator(args);
      } else if(transA && !transB) {
        CutlassGemmTNunfused gemm_operator;
        CutlassGemmTNunfused::Arguments args(
            {M, N, K},               // Gemm Problem dimensions
            {A, lda},                // Tensor-ref for source matrix A
            {B, ldb},                // Tensor-ref for source matrix B
            {(int32_t *)C, ldc},     // Tensor-ref for source matrix C
            {(int32_t *)C, ldc},     // Tensor-ref for destination matrix D (may be different memory
                                     // than source C matrix)
            {alpha_int, beta_int});  // Scalars used in the Epilogue
        return gemm_operator(args);
      } else if(!transA && transB) {
        CutlassGemmNTunfused gemm_operator;
        CutlassGemmNTunfused::Arguments args(
            {M, N, K},               // Gemm Problem dimensions
            {A, lda},                // Tensor-ref for source matrix A
            {B, ldb},                // Tensor-ref for source matrix B
            {(int32_t *)C, ldc},     // Tensor-ref for source matrix C
            {(int32_t *)C, ldc},     // Tensor-ref for destination matrix D (may be different memory
                                     // than source C matrix)
            {alpha_int, beta_int});  // Scalars used in the Epilogue
        return gemm_operator(args);
      } else {  // Final case (transA && transB)
        CutlassGemmTTunfused gemm_operator;
        CutlassGemmTTunfused::Arguments args(
            {M, N, K},               // Gemm Problem dimensions
            {A, lda},                // Tensor-ref for source matrix A
            {B, ldb},                // Tensor-ref for source matrix B
            {(int32_t *)C, ldc},     // Tensor-ref for source matrix C
            {(int32_t *)C, ldc},     // Tensor-ref for destination matrix D (may be different memory
                                     // than source C matrix)
            {alpha_int, beta_int});  // Scalars used in the Epilogue
        return gemm_operator(args);
      }
    }
  }
}

#include <iostream>
#include "cutlass/gemm/device/gemm.h"

//--------------------------------------------------------------------------------------------------

void fill_random(abtype *mtr, size_t sz) {
  const int bnd = 200;
  for(size_t i = 0; i < sz; i++) {
    mtr[i] = (abtype)(rand() % bnd - bnd / 2);
  }
}

void show_matrix(int8_t *mtr, int nrows, int ncols) {
  for(size_t i = 0; i < nrows; i++) {
    for(size_t j = 0; j < ncols; j++) {
      std::cout << " " << (int)mtr[i * ncols + j];
    }
    std::cout << std::endl;
  }
}

void show_float_matrix(float *mtr, int nrows, int ncols) {
  for(size_t i = 0; i < nrows; i++) {
    for(size_t j = 0; j < ncols; j++) {
      std::cout << " " << mtr[i * ncols + j];
    }
    std::cout << std::endl;
  }
}

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(int M,
                                     int N,
                                     int K,
                                     float alpha,
                                     abtype const *A,
                                     int lda,
                                     abtype const *B,
                                     int ldb,
                                     float beta,
                                     float *C,
                                     int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if(i < M && j < N) {
    float accumulator = 0;

    for(int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(int M,
                          int N,
                          int K,
                          float alpha,
                          abtype const *A,
                          int lda,
                          abtype const *B,
                          int ldb,
                          float beta,
                          float *C,
                          int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

void sanity_test() {
  int m = 4;
  int n = 4;
  int k = 1;

  int lda = m;
  int ldb = k;
  int ldc = m;

  // Host memory
  abtype *mtra;
  abtype *mtrb;
  float *mtrc;
  float *alpha;
  float *beta;
  float *bias;

  cudaHostAlloc(&mtra, sizeof(abtype) * m * k, cudaHostAllocMapped);
  cudaHostAlloc(&mtrb, sizeof(abtype) * k * n, cudaHostAllocMapped);
  cudaHostAlloc(&mtrc, sizeof(float) * m * n, cudaHostAllocMapped);
  cudaHostAlloc(&alpha, sizeof(float) * m * n, cudaHostAllocMapped);
  cudaHostAlloc(&beta, sizeof(float) * m * n, cudaHostAllocMapped);
  cudaHostAlloc(&bias, sizeof(float) * m * n, cudaHostAllocMapped);

  fill_random(mtra, sizeof(float) * m * k);
  fill_random(mtrb, sizeof(float) * k * n);
  //  fill_random_float(mtrc, sizeof(float) * m * n);
  for(size_t i = 0; i < m * n; i++) {
    alpha[i] = 1.0;
    beta[i] = 1.0;
    bias[i] = 0.0;
  }

  // Device memory
  abtype *mtra_dev;
  abtype *mtrb_dev;
  float *mtrc_dev;
  float *alpha_dev;
  float *beta_dev;
  float *bias_dev;

  cudaMalloc(&mtra_dev, sizeof(abtype) * m * k);
  cudaMemcpy(mtra_dev, mtra, sizeof(abtype) * m * k, cudaMemcpyHostToDevice);
  cudaMalloc(&mtrb_dev, sizeof(abtype) * k * n);
  cudaMemcpy(mtrb_dev, mtrb, sizeof(abtype) * k * n, cudaMemcpyHostToDevice);

  cudaMalloc(&mtrc_dev, sizeof(float) * m * n);

  cudaMalloc(&alpha_dev, sizeof(float) * m * n);
  cudaMemcpy(alpha_dev, alpha, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMalloc(&beta_dev, sizeof(float) * m * n);
  cudaMemcpy(beta_dev, beta, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMalloc(&bias_dev, sizeof(float) * m * n);
  cudaMemcpy(bias_dev, bias, sizeof(float) * m * n, cudaMemcpyHostToDevice);

//  cutlass_igemm_nn(0,
//                   0,
//                   m,
//                   n,
//                   k,
//                   alpha_dev,
//                   mtra_dev,
//                   lda,
//                   mtrb_dev,
//                   ldb,
//                   beta_dev,
//                   mtrc_dev,
//                   ldc,
//                   0, /*We want this to be true for best performance*/
//                   1, /* fused unquantisation (and bias addition (and activation function) if those
//                         are present). Should be true for best performance */
//                   bias_dev,
//                   0 /*doRelu*/);
  // Ref
  ReferenceGemm(m, n, k, 1.0, mtra_dev, lda, mtrb_dev, ldb, 0, mtrc_dev, ldc);

  cudaMemcpy(mtrc, mtrc_dev, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  //  std::cout <<"S:" <<(int)s <<std::endl;
  show_matrix(mtra, m, k);
  show_matrix(mtrb, k, n);
  show_float_matrix(mtrc, m, n);

  cudaFree(mtra);
  cudaFree(mtrb);
  cudaFree(mtrc);
  cudaFree(mtra_dev);
  cudaFree(mtrb_dev);
  cudaFree(mtrc_dev);
}

int main() {
  sanity_test();

  return 0;
}
