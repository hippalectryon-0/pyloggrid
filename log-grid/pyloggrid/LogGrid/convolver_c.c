#include <stdio.h>
#include <stdlib.h>
#ifdef _MSC_VER
    #include "Complex.h"  // Microsoft Complex.h
#else
    #include <stdint.h>
    #include <complex.h>
#endif
#include <omp.h>

#ifdef _MSC_VER  // Replace C99 with MSVC "equivalents" or mockups
    #define always_inline __forceinline
    #define __builtin_unreachable() __assume(0)
    #define restrict __restrict
    typedef _Dcomplex Complex;
#else
    #define always_inline inline __attribute__((always_inline))
    typedef double _Complex Complex;
#endif

static always_inline Complex ComplexMult(Complex a, Complex b) {
    Complex result;
    #ifdef _MSC_VER
        result = _Cmulcc(a, b);
    #else
        result = a * b;
    #endif
    return result;
}

static always_inline Complex ComplexAddMSVC(Complex a, Complex b) {
    return _Cbuild(creal(a) + creal(b), cimag(a)+cimag(b));
}

static always_inline void convolve_inner(const size_t i, const uint32_t *restrict kernel, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    Complex c1 = f[kernel[i + 1]];
    Complex c2 = g[kernel[i + 2]];

    Complex x;
    switch (kernel[i + 3]) {
        case 0:
            x = ComplexMult(c1, c2);
            break;
        case 1:
            x = ComplexMult(conj(c1), c2);
            break;
        case 2:
            x = ComplexMult(c1, conj(c2));
            break;
        default:
            __builtin_unreachable();
    }

    #ifdef _MSC_VER
        arr_out[kernel[i]] = ComplexAddMSVC(arr_out[kernel[i]], x);
    #else
        arr_out[kernel[i]] += x;
    #endif
}

void convolve(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    for (size_t i = 0; i < kernel_N; i += 4) {
        convolve_inner(i, kernel, f, g, arr_out);
    }
}

void convolve_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    int i;  // can't use <for(int i ...> because of MSVC
    #pragma omp parallel for schedule(static)
    for (i = 0; i < kernel_N; i += 4) {
        convolve_inner(i, kernel, f, g, arr_out);
    }
}

void convolve_list_omp(const uint32_t *kernel, const uint32_t kernel_N, const Complex **f_list, const Complex **g_list, const uint32_t f_size, const uint32_t N_batch, Complex *arr_out) {
    for (size_t i_batch = 0; i_batch < N_batch; i_batch++) {
        int i;  // can't use <for(int i ...> because of MSVC
        #pragma omp parallel for schedule(static)
        for (i = 0; i < kernel_N; i += 4) {
            convolve_inner(i, kernel, f_list[i_batch], g_list[i_batch], &arr_out[i_batch *f_size]);
        }
    }
}

static inline always_inline void convolve_list_batch_V_inner(const size_t i_batch, const size_t i, const uint32_t *restrict kernel, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, Complex *restrict arr_out, const size_t V) {
    #ifdef _MSC_VER
        Complex* c1 = (Complex*)malloc(V * sizeof(Complex));
        Complex* c2 = (Complex*)malloc(V * sizeof(Complex));
        Complex* x = (Complex*)malloc(V * sizeof(Complex));
    #else
        Complex c1[V];
        Complex c2[V];
        Complex x[V];
    #endif


    for (size_t j=0; j<V; j++) {
        c1[j] = f_list[i_batch + j][kernel[i + 1]];
        c2[j] = g_list[i_batch + j][kernel[i + 2]];
    }

    switch (kernel[i + 3]) {
        case 0:
            for (size_t j=0; j<V; j++) {
                x[j] = ComplexMult(c1[j], c2[j]);
            }
            break;
        case 1:
            for (size_t j=0; j<V; j++) {
                x[j] = ComplexMult(conj(c1[j]), c2[j]);
            }
            break;
        case 2:
            for (size_t j=0; j<V; j++) {
                x[j] = ComplexMult(c1[j], conj(c2[j]));
            }
            break;
        default:
            __builtin_unreachable();
    }

    for (size_t j=0; j<V; j++) {
        #ifdef _MSC_VER
            arr_out[(i_batch + j) * f_size + kernel[i]] = ComplexAddMSVC(arr_out[(i_batch + j) * f_size + kernel[i]], x[j]);
        #else
            arr_out[(i_batch + j) * f_size + kernel[i]] += x[j];
        #endif
    }

    #ifdef _MSC_VER
        free(c1);
        free(c2);
        free(x);
    #endif
}

static inline always_inline void convolve_list_batch_V(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out, const size_t V) {
    if (N_batch % V != 0) { __builtin_unreachable(); }

    for (size_t i_batch = 0; i_batch < N_batch; i_batch+=V) {
        for (size_t i = 0; i < kernel_N; i += 4) {
            convolve_list_batch_V_inner(i_batch, i, kernel, f_list, g_list, f_size, arr_out, V);
        }
    }
}

static inline always_inline void convolve_list_batch_V_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out, const size_t V) {
    if (N_batch % V != 0) { __builtin_unreachable(); }

    int i_batch;  // can't use <for(int i ...> because of MSVC
    #pragma omp parallel for schedule(static) collapse(2)
    for (i_batch = 0; i_batch < N_batch; i_batch+=V) {
        for (size_t i = 0; i < kernel_N; i += 4) {
            convolve_list_batch_V_inner(i_batch, i, kernel, f_list, g_list, f_size, arr_out, V);
        }
    }
}

void convolve_list_batch_V2(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 2;
    convolve_list_batch_V(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void convolve_list_batch_V3(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 3;
    convolve_list_batch_V(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void convolve_list_batch_V4(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 4;
    convolve_list_batch_V(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void convolve_list_batch_V2_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 2;
    convolve_list_batch_V_omp(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void convolve_list_batch_V3_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 3;
    convolve_list_batch_V_omp(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void convolve_list_batch_V4_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out) {
    const size_t V = 4;
    convolve_list_batch_V_omp(kernel, kernel_N, f_list, g_list, f_size, N_batch, arr_out, V);
}

void set_omp_threads(const uint32_t N_threads) {
    #ifdef USE_OMP
        omp_set_num_threads(N_threads);
    #endif
}
