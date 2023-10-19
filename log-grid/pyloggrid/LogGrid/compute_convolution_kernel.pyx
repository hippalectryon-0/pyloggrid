# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, infer_types=True
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t

# python setup_linux.py build_ext --inplace

cdef inline short offset_coord(const short pos, const short offset, const Py_ssize_t N_pts, const bint is_axis_0, const bint has_k0) nogil:
    """offset a coord, returns the offset coord or -1 if out of bounds"""
    cdef short coord
    if is_axis_0:
        coord = pos + offset
        if has_k0:
            oob = not((0 < coord < N_pts) or (pos==0 and coord==0))
        else:
            oob = not (0 <= coord < N_pts)
    else:
        n = N_pts//2
        if pos < n:
            coord = pos - offset
            oob = not (0 <= coord < n)
        else:
            coord = pos + offset
            if has_k0:
                oob = not ((n < coord < N_pts) or (pos == n and coord == n))
            else:
                oob = not (n <= coord < N_pts)
    if oob:
        coord = -1

    return coord

cdef inline short flip_coord(short coord, Py_ssize_t N_pts, bint has_k0) nogil:
    """flip a coordinate"""
    cdef short n, n_k0
    n = <short>(N_pts // 2)
    n_k0 = (0 if has_k0 else 1)
    if coord < n:
        coord = n + n - coord - n_k0
    else:
        coord = n - (coord - n + n_k0)
    return coord

cdef inline Py_ssize_t midx_2d(const short a, const short b, const short ny) nogil:
    """3D to 1D coord"""
    return (a * ny) + b

cdef inline Py_ssize_t midx_3d(const short a, const short b, const short c, const short ny, const short nz) nogil:
    """3D to 1D coord"""
    return (a * nz + b) * ny + c

cdef np.ndarray compute_interaction_kernel_1D_c(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, const bint has_k0, const short sx):
    cdef short sign_x0, sign_x1, coord_x0, coord_x1
    cdef uint32_t kernel_i = -1
    cdef short[:, ::1] offsets_i_v, signs_i_v
    kernel_size = sx * offsets_v.shape[0]
    if has_k0:
        kernel_size *= 2
    cdef np.ndarray kernel = np.zeros((kernel_size, 4), dtype=np.uint32)
    cdef uint32_t[:, ::1] kernel_v = kernel
    for i_nx in range(sx):
        if has_k0 and i_nx == 0:
            offsets_i_v = offsets_k0_v
            signs_i_v = signs_k0_v
        else:
            offsets_i_v = offsets_v
            signs_i_v = signs_v
        for i_kx in range(offsets_i_v.shape[0]):
            sign_x0, sign_x1 = signs_i_v[i_kx][0], signs_i_v[i_kx][1]
            coord_x0 = offset_coord(i_nx, offsets_i_v[i_kx][0], sx, True, has_k0)
            coord_x1 = offset_coord(i_nx, offsets_i_v[i_kx][1], sx, True, has_k0)
            if coord_x0 < 0 or coord_x1 < 0:  # check that both iteractions are in the grid
                continue
            kernel_i += 1

            kernel_v[kernel_i, 0] = i_nx
            if sign_x0 == -1:
                kernel_v[kernel_i, 1] = coord_x0
                kernel_v[kernel_i, 2] = coord_x1
                kernel_v[kernel_i, 3] = 1  # 1: f.c * g
            elif sign_x1 == -1:
                kernel_v[kernel_i, 1] = coord_x0
                kernel_v[kernel_i, 2] = coord_x1
                kernel_v[kernel_i, 3] = 2  # 2: f * g.c
            else:
                kernel_v[kernel_i, 1] = coord_x0 * sign_x0
                kernel_v[kernel_i, 2] = coord_x1 * sign_x1
                kernel_v[kernel_i, 3] = 0  # 0: f * g
    return kernel[:kernel_i+1]

cpdef np.ndarray compute_interaction_kernel_1D(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, const bint has_k0, const short sx):
    return compute_interaction_kernel_1D_c(offsets_v, signs_v, offsets_k0_v, signs_k0_v, has_k0, sx).flatten()

cdef np.ndarray compute_interaction_kernel_2D_c(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, short[:, ::1] offsets_k1_v, short[:, ::1] signs_k1_v, const bint has_k0, const short sx, const short sy):
    cdef short sign_x0, sign_x1, sign_y0, sign_y1, coord_x0, coord_x1, coord_y0, coord_y1
    cdef uint32_t kernel_i = -1
    cdef short[:, ::1] offsets_x_v, offsets_y_v, signs_x_v, signs_y_v
    kernel_size = sx * sy * offsets_v.shape[0] **2
    if has_k0:
        kernel_size *= 2
    cdef np.ndarray kernel = np.zeros((kernel_size, 4), dtype=np.uint32)
    cdef uint32_t[:, ::1] kernel_v = kernel
    for i_nx in range(sx):
        if has_k0 and i_nx == 0:
            offsets_x_v = offsets_k0_v
            signs_x_v = signs_k0_v
        else:
            offsets_x_v = offsets_v
            signs_x_v = signs_v
        for i_kx in range(offsets_x_v.shape[0]):
            sign_x0, sign_x1 = signs_x_v[i_kx][0], signs_x_v[i_kx][1]
            coord_x0 = offset_coord(i_nx, offsets_x_v[i_kx][0], sx, True, has_k0)
            coord_x1 = offset_coord(i_nx, offsets_x_v[i_kx][1], sx, True, has_k0)
            if coord_x0 < 0 or coord_x1 < 0:  # check that both iteractions are in the grid
                continue

            for i_ny in range(sy):  # Y
                if has_k0 and i_ny == sy//2:
                    offsets_y_v = offsets_k1_v
                    signs_y_v = signs_k1_v
                else:
                    offsets_y_v = offsets_v
                    signs_y_v = signs_v
                for i_ky in range(offsets_y_v.shape[0]):
                    sign_y0, sign_y1 = signs_y_v[i_ky][0], signs_y_v[i_ky][1]
                    coord_y0 = offset_coord(i_ny, offsets_y_v[i_ky][0], sy , False, has_k0)
                    coord_y1 = offset_coord(i_ny, offsets_y_v[i_ky][1], sy, False, has_k0)
                    if coord_y0 < 0 or coord_y1 < 0:  # check that both iteractions are in the grid
                        continue
                    kernel_i += 1

                    # remap
                    if sign_y0 == -1:
                        coord_y0 = flip_coord(coord_y0, sy, has_k0)
                    elif sign_y1 == -1:
                        coord_y1 = flip_coord(coord_y1, sy, has_k0)
                    elif sign_y0 == 0:
                        coord_y0 = int(sy // 2)
                    elif sign_y1 == 0:
                        coord_y1 = int(sy // 2)

                    kernel_v[kernel_i, 0] = midx_2d(i_nx, i_ny, sy)
                    if sign_x0 == -1:
                        kernel_v[kernel_i, 1] = midx_2d(coord_x0, flip_coord(coord_y0, sy, has_k0), sy)
                        kernel_v[kernel_i, 2] = midx_2d(coord_x1, coord_y1, sy)
                        kernel_v[kernel_i, 3] = 1  # 1: f.c * g
                    elif sign_x1 == -1:
                        kernel_v[kernel_i, 1] = midx_2d(coord_x0, coord_y0, sy)
                        kernel_v[kernel_i, 2] = midx_2d(coord_x1, flip_coord(coord_y1, sy, has_k0), sy)
                        kernel_v[kernel_i, 3] = 2  # 2: f * g.c
                    else:
                        kernel_v[kernel_i, 1] = midx_2d(coord_x0 * sign_x0, coord_y0, sy)
                        kernel_v[kernel_i, 2] = midx_2d(coord_x1 * sign_x1, coord_y1, sy)
                        kernel_v[kernel_i, 3] = 0  # 0: f * g
    return kernel[:kernel_i+1]

cpdef np.ndarray compute_interaction_kernel_2D(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, short[:, ::1] offsets_k1_v, short[:, ::1] signs_k1_v, const bint has_k0, const short sx, const short sy):
    return compute_interaction_kernel_2D_c(offsets_v, signs_v, offsets_k0_v, signs_k0_v, offsets_k1_v, signs_k1_v, has_k0, sx, sy).flatten()

cdef np.ndarray compute_interaction_kernel_3D_c(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, short[:, ::1] offsets_k1_v, short[:, ::1] signs_k1_v, const bint has_k0, const short sx, const short sy, const short sz):
    cdef short sign_x0, sign_x1, sign_y0, sign_y1, sign_z0, sign_z1, coord_x0, coord_x1, coord_y0, coord_y1, coord_z0, coord_z1
    cdef short[:, ::1] offsets_x_v, offsets_y_v, offsets_z_v, signs_x_v, signs_y_v, signs_z_v
    cdef uint32_t kernel_i = -1
    kernel_size = sx * sy * sz * offsets_v.shape[0] ** 3
    if has_k0:
        kernel_size *= 2
    cdef np.ndarray kernel = np.zeros((kernel_size, 4), dtype=np.uint32)
    cdef uint32_t[:, ::1] kernel_v = kernel
    for i_nx in range(sx):
        if has_k0 and i_nx == 0:
            offsets_x_v = offsets_k0_v
            signs_x_v = signs_k0_v
        else:
            offsets_x_v = offsets_v
            signs_x_v = signs_v
        for i_kx in range(offsets_x_v.shape[0]):
            sign_x0, sign_x1 = signs_x_v[i_kx][0], signs_x_v[i_kx][1]
            coord_x0 = offset_coord(i_nx, offsets_x_v[i_kx][0], sx, True, has_k0)
            coord_x1 = offset_coord(i_nx, offsets_x_v[i_kx][1], sx, True, has_k0)
            if coord_x0 < 0 or coord_x1 < 0:  # check that both iteractions are in the grid
                continue

            for i_ny in range(sy):  # Y
                if has_k0 and i_ny == sy//2:
                    offsets_y_v = offsets_k1_v
                    signs_y_v = signs_k1_v
                else:
                    offsets_y_v = offsets_v
                    signs_y_v = signs_v
                for i_ky in range(offsets_y_v.shape[0]):
                    sign_y0, sign_y1 = signs_y_v[i_ky][0], signs_y_v[i_ky][1]
                    coord_y0 = offset_coord(i_ny, offsets_y_v[i_ky][0], sy , False, has_k0)
                    coord_y1 = offset_coord(i_ny, offsets_y_v[i_ky][1], sy, False, has_k0)
                    if coord_y0 < 0 or coord_y1 < 0:  # check that both iteractions are in the grid
                        continue

                    # remap
                    if sign_y0 == -1:
                        coord_y0 = flip_coord(coord_y0, sy, has_k0)
                    elif sign_y1 == -1:
                        coord_y1 = flip_coord(coord_y1, sy, has_k0)
                    elif sign_y0 == 0:
                        coord_y0 = int(sy // 2)
                    elif sign_y1 == 0:
                        coord_y1 = int(sy // 2)

                    for i_nz in range(sz):  # Z
                        if has_k0 and i_nz == sz // 2:
                            offsets_z_v = offsets_k1_v
                            signs_z_v = signs_k1_v
                        else:
                            offsets_z_v = offsets_v
                            signs_z_v = signs_v
                        for i_kz in range(offsets_z_v.shape[0]):
                            sign_z0, sign_z1 = signs_z_v[i_kz][0], signs_z_v[i_kz][1]
                            coord_z0 = offset_coord(i_nz, offsets_z_v[i_kz][0], sz, False, has_k0)
                            coord_z1 = offset_coord(i_nz, offsets_z_v[i_kz][1], sz, False, has_k0)
                            if coord_z0 < 0 or coord_z1 < 0:  # check that both iteractions are in the grid
                                continue
                            kernel_i += 1

                            # remap
                            if sign_z0 == -1:
                                coord_z0 = flip_coord(coord_z0, sy, has_k0)
                            elif sign_z1 == -1:
                                coord_z1 = flip_coord(coord_z1, sy, has_k0)
                            elif sign_z0 == 0:
                                coord_z0 = int(sz // 2)
                            elif sign_z1 == 0:
                                coord_z1 = int(sz // 2)

                            kernel_v[kernel_i,0] = midx_3d(i_nx, i_ny, i_nz, sy, sz)
                            if sign_x0 == -1:
                                kernel_v[kernel_i, 1] = midx_3d(coord_x0, flip_coord(coord_y0, sy, has_k0), flip_coord(coord_z0, sz, has_k0), sy, sz)
                                kernel_v[kernel_i, 2] = midx_3d(coord_x1, coord_y1, coord_z1, sy, sz)
                                kernel_v[kernel_i, 3] = 1  # 1: f.c * g
                            elif sign_x1 == -1:
                                kernel_v[kernel_i, 1] = midx_3d(coord_x0, coord_y0, coord_z0, sy, sz)
                                kernel_v[kernel_i, 2] = midx_3d(coord_x1, flip_coord(coord_y1, sy, has_k0), flip_coord(coord_z1, sz, has_k0), sy, sz)
                                kernel_v[kernel_i, 3] = 2  # 2: f * g.c
                            else:
                                kernel_v[kernel_i, 1] = midx_3d(coord_x0 * sign_x0, coord_y0, coord_z0, sy, sz)
                                kernel_v[kernel_i, 2] = midx_3d(coord_x1 * sign_x1, coord_y1, coord_z1, sy, sz)
                                kernel_v[kernel_i, 3] = 0  # 0: f * g

    return kernel[:kernel_i+1]

cpdef np.ndarray compute_interaction_kernel_3D(short[:, ::1] offsets_v, short[:, ::1] signs_v, short[:, ::1] offsets_k0_v, short[:, ::1] signs_k0_v, short[:, ::1] offsets_k1_v, short[:, ::1] signs_k1_v, const bint has_k0, const short sx, const short sy, const short sz):
    return compute_interaction_kernel_3D_c(offsets_v, signs_v, offsets_k0_v, signs_k0_v, offsets_k1_v, signs_k1_v, has_k0, sx, sy, sz).flatten()
