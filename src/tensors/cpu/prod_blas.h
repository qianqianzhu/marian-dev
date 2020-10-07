#if MKL_FOUND
#include <mkl.h>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
#endif

#if 1 //@TODO proper guards
#include <dnnl.h>
#endif

inline void sgemm_dnnl(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc) {
  dnnl_sgemm(transA ? 'T' : 'N', 
             transB ? 'T' : 'N', 
             (dnnl_dim_t)rows_a,
             (dnnl_dim_t)rows_b, 
             (dnnl_dim_t)width, 
             alpha, 
             a, 
             (dnnl_dim_t)lda,
             b, 
             (dnnl_dim_t)ldb, 
             beta, 
             c, 
             (dnnl_dim_t)ldc);
}


inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc,
                  bool useDNNL=false) {
#if 1
  sgemm_dnnl(transA,
             transB,
             rows_a,
             rows_b,
             width,
             alpha,
             a,
             lda,
             b,
             ldb,
             beta,
             c,
             ldc);
#else
  #if BLAS_FOUND
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                rows_a,
                rows_b,
                width,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc);
  #else
      transA; transB; rows_a; rows_b; width; alpha; a; lda; b; ldb; beta; c; ldc; // make compiler happy
      ABORT("Marian must be compiled with a BLAS library");
  #endif
#endif
}
