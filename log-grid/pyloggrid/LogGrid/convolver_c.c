// https://github.com/chcomin/ctypes-numpy-example
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdint.h>
#include <omp.h>

typedef double _Complex Complex;

static inline __attribute__((always_inline)) void convolve_inner(const size_t i, const uint32_t *restrict kernel, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    Complex c1 = f[kernel[i + 1]];
    Complex c2 = g[kernel[i + 2]];

    Complex x;
    switch (kernel[i + 3]) {
        case 0:
            x = c1 * c2;
            break;
        case 1:
            x = conj(c1) *c2;
            break;
        case 2:
            x = c1* conj(c2);
            break;
        default:
            __builtin_unreachable();
    }

    arr_out[kernel[i]] += x;
}

void convolve(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    for (size_t i = 0; i < kernel_N; i += 4) {
        convolve_inner(i, kernel, f, g, arr_out);
    }
}

void convolve_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex *restrict f, const Complex *restrict g, Complex *restrict arr_out) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < kernel_N; i += 4) {
        convolve_inner(i, kernel, f, g, arr_out);
    }
}

void convolve_list_omp(const uint32_t *kernel, const uint32_t kernel_N, const Complex **f_list, const Complex **g_list, const uint32_t f_size, const uint32_t N_batch, Complex *arr_out) {
    for (size_t i_batch = 0; i_batch < N_batch; i_batch++) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < kernel_N; i += 4) {
            convolve_inner(i, kernel, f_list[i_batch], g_list[i_batch], &arr_out[i_batch *f_size]);
        }
    }
}

static inline __attribute__((always_inline)) void convolve_list_batch_V_inner(const size_t i_batch, const size_t i, const uint32_t *restrict kernel, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, Complex *restrict arr_out, const size_t V) {
    Complex c1[V];
    Complex c2[V];

    for (size_t j=0; j<V; j++) {
        c1[j] = f_list[i_batch + j][kernel[i + 1]];
        c2[j] = g_list[i_batch + j][kernel[i + 2]];
    }

    Complex x[V];
    switch (kernel[i + 3]) {
        case 0:
            for (size_t j=0; j<V; j++) {
                x[j] = c1[j] * c2[j];
            }
            break;
        case 1:
            for (size_t j=0; j<V; j++) {
                x[j] = conj(c1[j]) * c2[j];
            }
            break;
        case 2:
            for (size_t j=0; j<V; j++) {
                x[j] = c1[j] * conj(c2[j]);
            }
            break;
        default:
            __builtin_unreachable();
    }

    for (size_t j=0; j<V; j++) {
        arr_out[(i_batch + j) * f_size + kernel[i]] += x[j];
    }
}

static inline __attribute__((always_inline)) void convolve_list_batch_V(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out, const size_t V) {
    if (N_batch % V != 0) { __builtin_unreachable(); }

    for (size_t i_batch = 0; i_batch < N_batch; i_batch+=V) {
        for (size_t i = 0; i < kernel_N; i += 4) {
            convolve_list_batch_V_inner(i_batch, i, kernel, f_list, g_list, f_size, arr_out, V);
        }
    }
}

static inline __attribute__((always_inline)) void convolve_list_batch_V_omp(const uint32_t *restrict kernel, const uint32_t kernel_N, const Complex **restrict f_list, const Complex **restrict g_list, const uint32_t f_size, const uint32_t N_batch, Complex *restrict arr_out, const size_t V) {
    if (N_batch % V != 0) { __builtin_unreachable(); }

    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i_batch = 0; i_batch < N_batch; i_batch+=V) {
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
