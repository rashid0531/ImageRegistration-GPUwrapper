import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import numpy C/C++ API
from cython.operator cimport dereference

def cudaWarpPerspectiveWrapper(np.ndarray[np.float32_t, ndim=3] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags=INTER_NEAREST):
    
    #mrc689
    np.import_array()
    # Create GPU/device InputArray for src
    cdef Mat src_mat
    cdef GpuMat src_gpu
    pyopencv_to(<PyObject*> _src, src_mat)
    #cdef np.ndarray yo = <np.ndarray> pyopencv_from(src_mat)
    #print(yo)
    src_gpu.upload(src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    pyopencv_to(<PyObject*> _M, M_mat)

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])
    # Create empty GPU/device OutputArray for dst
    cdef GpuMat dst_gpu = GpuMat()
    warpPerspective(src_gpu, dst_gpu, M_mat, size, INTER_NEAREST)
    #lets see what happens
    cdef Algorithm a = Algorithm()
    # Get result of dst
    cdef Mat dst_host
    dst_gpu.download(dst_host)
    cdef np.ndarray out = <np.ndarray> pyopencv_from(dst_host)
    return out

def match_feature(np.ndarray[np.float32_t, ndim=3] _src,
                               np.ndarray[np.float32_t, ndim=2] _M):

    np.import_array()
    # Create GPU/device InputArray for src
    cdef Mat src_mat
    cdef GpuMat src_gpu
    pyopencv_to(<PyObject*> _src, src_mat)
    src_gpu.upload(src_mat)

    cdef Mat src_mat_2
    cdef GpuMat src_gpu_2
    pyopencv_to(<PyObject*> _M, src_mat_2)
    src_gpu_2.upload(src_mat_2)
    
    cdef Ptr[DescriptorMatcher] matcher = DescriptorMatcher.createBFMatcher(4)
    cdef vector[vector[DMatch]] matches
    dereference(matcher).knnMatch(src_gpu,src_gpu_2,matches,2)
    #print(macthes)
    #matcher.knnMatch(src_gpu,src_gpu_2,matches,2)
    #cdef np.ndarray out = <np.ndarray> pyopencv_from(matches)
    print("no problem so far")
    #return out
