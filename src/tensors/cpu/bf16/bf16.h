/* Adapted from https://gist.github.com/emfomenk/4bdc70908cc5c30ad2a97e5030a22eaf */
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>
#include <type_traits>

#include "bfloat16.hpp"
#include <dnnl.hpp>


namespace proxy {

// Read from a float vector, write to DNNL bfloat16
inline void float2bf16dnnlmemory(const float *handle, dnnl::memory &mem, size_t items) {
    uint16_t *dst = static_cast<uint16_t *>(mem.get_data_handle());
    for (size_t i = 0; i < items; ++i) {
        dst[i] = dnnl::impl::bfloat16_t{handle[i]}.raw_bits_;
    }
}

inline void float2bfloat16(float * input, size_t items) {
    std::vector<float> tmp(items);
    std::memcpy(tmp.data(), input, items*sizeof(float));
    uint16_t *dst = reinterpret_cast<uint16_t *>(input);
    for (size_t i = 0; i < items; i++) {
        dst[i] = dnnl::impl::bfloat16_t{tmp[i]}.raw_bits_;
    }
}

inline void bfloat162float(void * input, size_t items) {
    std::vector<uint16_t> tmp(items);
    std::memcpy(tmp.data(), input, items*sizeof(uint16_t));
    float *dst = reinterpret_cast<float *>(input);
    for (size_t i = 0; i < items; i++) {
        dnnl::impl::bfloat16_t res;
        res.raw_bits_ = tmp[i];
        dst[i] = (float)res;
    }
}

inline void printMem(dnnl::memory &mem, size_t items, bool isBF16=false) {
    if (isBF16) {
        uint16_t *dst = static_cast<uint16_t *>(mem.get_data_handle());
        for (size_t i = 0; i < items; ++i) {
            dnnl::impl::bfloat16_t res;
            res.raw_bits_ = dst[i];
            std::cerr << (float)res << " ";
        }
        std::cerr << std::endl;
    } else {
        float *dst = static_cast<float *>(mem.get_data_handle());
        for (size_t i = 0; i < items; ++i) {
            std::cerr << dst[i] << " ";
        }
        std::cerr << std::endl;
    }
}

template <typename c_dt, bool beta_is_zero = true>
dnnl::status gemm_bf16bf16(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const void *A, dnnl_dim_t lda,
        const void *B, dnnl_dim_t ldb, float beta, c_dt *C, dnnl_dim_t ldc) {
    using namespace dnnl;
    using dims = memory::dims;

    if ((int)get_effective_cpu_isa() < (int)cpu_isa::avx512_core)
        return status::unimplemented;

    static engine eng;
    static matmul matmul_p;
    static std::once_flag initialized;

    static_assert(std::is_same<c_dt, float>::value ||
            std::is_same<c_dt, void>::value, "expect float or void (bf16)");
    constexpr memory::data_type bf16 = memory::data_type::bf16;
    constexpr memory::data_type c_data_type =
        std::is_same<c_dt, float>::value ? memory::data_type::f32 : bf16;

    std::call_once(initialized, [=] {
        eng = engine(engine::kind::cpu, 0);

        memory::dims rt_rt_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        memory::dims rt_1_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        memory::desc ab_md(rt_rt_dims, bf16, rt_rt_dims);
        memory::desc c_md(rt_rt_dims, c_data_type, rt_1_dims);

        primitive_attr attr;
        attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
        if (beta != 0.f) {
            assert(beta == 1.f); // current limitation
            post_ops po;
            po.append_sum(beta);
            attr.set_post_ops(po);
        }

        matmul::desc matmul_d(ab_md, ab_md, c_md);
        matmul::primitive_desc matmul_pd(matmul_d, attr, eng, true);
        if (matmul_pd) matmul_p = matmul(matmul_pd);
    });

    bool ok = (bool)matmul_p
        && (!beta_is_zero || beta == 0.f)
        && (!!beta_is_zero || beta == 1.f);
    if (!ok) return status::runtime_error;

    dims a_strides = tolower(transa) == 'n' ? dims {lda, 1} : dims {1, lda};
    dims b_strides = tolower(transb) == 'n' ? dims {ldb, 1} : dims {1, ldb};

    memory A_m({{M, K}, bf16, a_strides}, eng, (void *)A);
    memory B_m({{K, N}, bf16, b_strides}, eng, (void *)B);
    memory C_m({{M, N}, c_data_type, {ldc, 1}}, eng, (void *)C);

    // Prepare oneDNN memory for alpha
    memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

    stream s(eng);
    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_m}, {DNNL_ARG_WEIGHTS, B_m}, {DNNL_ARG_DST, C_m},
                    {DNNL_ARG_ATTR_OUTPUT_SCALES, alpha_m}});
    s.wait();

    return status::success;
}

template <typename c_dt, bool beta_is_zero>
dnnl::status gemm_f32f32bf16(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, c_dt *C, dnnl_dim_t ldc) {
    using namespace dnnl;
    using dims = memory::dims;

    if ((int)get_effective_cpu_isa() < (int)cpu_isa::avx512_core)
        return status::unimplemented;

    static engine eng;
    static engine eng_beta1;
    static matmul matmul_p;
    static matmul matmul_p_beta1;
    static std::once_flag initialized;

    static_assert(std::is_same<c_dt, float>::value ||
            std::is_same<c_dt, void>::value, "expect float or void (bf16)");
    constexpr memory::data_type bf16 = memory::data_type::bf16;
    constexpr memory::data_type c_data_type =
        std::is_same<c_dt, float>::value ? memory::data_type::f32 : bf16;

    std::call_once(initialized, [=] {
        eng = engine(engine::kind::cpu, 0);

        memory::dims rt_rt_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        memory::dims rt_1_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        memory::desc ab_md(rt_rt_dims, bf16, rt_rt_dims);
        memory::desc c_md(rt_rt_dims, c_data_type, rt_1_dims);

        primitive_attr attr;
        attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
        if (beta != 0.f) {
            assert(beta == 1.f); // current limitation
            post_ops po;
            po.append_sum(beta);
            attr.set_post_ops(po);
        }

        matmul::desc matmul_d(ab_md, ab_md, c_md);
        matmul::primitive_desc matmul_pd(matmul_d, attr, eng, true);
        if (matmul_pd) matmul_p = matmul(matmul_pd);
    });

    dims a_strides = tolower(transa) == 'n' ? dims {lda, 1} : dims {1, lda};
    dims b_strides = tolower(transb) == 'n' ? dims {ldb, 1} : dims {1, ldb};


    // Init bf16 memory and convert to floats
    memory A_m({{M, K}, bf16, a_strides}, eng);
    float2bf16dnnlmemory(A, A_m, (size_t)M*(size_t)K);
    memory B_m({{K, N}, bf16, b_strides}, eng);
    float2bf16dnnlmemory(B, B_m, (size_t)K*(size_t)N);
    memory C_m({{M, N}, c_data_type, {ldc, 1}}, eng, (void *)C);

    // Prepare oneDNN memory for alpha
    memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

    stream s(eng);
    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_m}, {DNNL_ARG_WEIGHTS, B_m}, {DNNL_ARG_DST, C_m},
                    {DNNL_ARG_ATTR_OUTPUT_SCALES, alpha_m}});
    s.wait();

    return status::success;
}

}

inline dnnl::status gemm_bf16bf16bf16(char transa, char transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const void *A, dnnl_dim_t lda,
        const void *B, dnnl_dim_t ldb, void *C, dnnl_dim_t ldc) {
    return proxy::gemm_bf16bf16<void>(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, 0, C, ldc);
}

inline dnnl::status gemm_bf16bf16f32(char transa, char transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const void *A, dnnl_dim_t lda,
        const void *B, dnnl_dim_t ldb, float *C, dnnl_dim_t ldc) {
    return proxy::gemm_bf16bf16<float>(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, 0, C, ldc);
}

inline dnnl::status gemm_f32f32bf16f32(bool transa, bool transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc) {
            if (beta == 0.f) {
                return proxy::gemm_f32f32bf16<float, true>(
            transa ? 'T' : 'N', transb ? 'T' : 'N', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            } else if (beta == 1.f) {
                return proxy::gemm_f32f32bf16<float, false>(
            transa ? 'T' : 'N', transb ? 'T' : 'N', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                assert(false); // We should not here
            }
}

inline dnnl::status gemm_f32f32bf16bf16(bool transa, bool transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, void *C, dnnl_dim_t ldc) {
            if (beta == 0.f) {
                return proxy::gemm_f32f32bf16<void, true>(
                        transa ? 'T' : 'N', transb ? 'T' : 'N', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            } else if (beta == 1.f) {
                return proxy::gemm_f32f32bf16<void, false>(
                        transa ? 'T' : 'N', transb ? 'T' : 'N', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                assert(false);
            }
}