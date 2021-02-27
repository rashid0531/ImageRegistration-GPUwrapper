from libcpp cimport bool
from cpython.ref cimport PyObject
from libcpp.vector cimport vector

# References PyObject to OpenCV object conversion code borrowed from OpenCV's own conversion file, cv2.cpp
cdef extern from 'pyopencv_converter.cpp':
    #mrc689 April 20,2017
    void import_array()
    cdef PyObject* pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject* o, Mat& m)
    #@staticmethod
    #cdef PyObject* pyopencv_from_generic_vec(const vector[vector[DMatch]]& value) 

cdef extern from 'opencv2/imgproc.hpp' namespace 'cv':
    cdef enum InterpolationFlags:
        INTER_NEAREST = 0
    cdef enum ColorConversionCodes:
        COLOR_BGR2GRAY

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1
    cdef int CV_32FC1

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Size_[T]:
        Size_() except +
        Size_(T width, T height) except +
        T width
        T height
    ctypedef Size_[int] Size2i
    ctypedef Size2i Size
    cdef cppclass Scalar[T]:
        Scalar() except +
        Scalar(T v0) except +

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int) except +
        void* data
        int rows
        int cols

    #added to test the Algorithm class inside core.hpp on May5th 12.52 AM.
    cdef cppclass Algorithm:
        Algorithm() except +
        
cdef extern from 'opencv2/core/base.hpp' namespace 'cv':
    cdef enum NormTypes:
        NORM_INF= 1,
        NORM_L1= 2,
        NORM_L2= 4,
        NORM_HAMMING= 6,
        NORM_HAMMING2= 7,

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        void upload(Mat arr) except +
        void download(Mat dst) const
    cdef cppclass Stream:
        Stream() except +

cdef extern from 'opencv2/core/types.hpp' namespace 'cv':
    cdef cppclass DMatch:
        DMatch() except +
        float distance
        int imgIdx
        int queryIdx
        int trainIdx

cdef extern from 'opencv2/core/cvstd.hpp' namespace 'cv':
    cdef cppclass Ptr[T]:
        Ptr() except +
        Ptr(Ptr*) except +
        T& operator* () # probably no exceptions	 

cdef extern from 'opencv2/cudafeatures2d.hpp' namespace 'cv::cuda':
    cdef cppclass DescriptorMatcher:
        @staticmethod
        Ptr[DescriptorMatcher] createBFMatcher(int normType) except+
        #Expected to see error here
        void knnMatch(GpuMat queryDescriptors, GpuMat trainDescriptors, vector[vector[DMatch]] &matches,int k)

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef int getCudaEnabledDeviceCount()
    cdef void setDevice(int device)
    cdef int getDevice()

cdef extern from 'opencv2/cudawarping.hpp' namespace 'cv::cuda':
    cdef void warpPerspective(GpuMat src, GpuMat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue, Stream& stream)
    # Function using default values
    cdef void warpPerspective(GpuMat src, GpuMat dst, Mat M, Size dsize, int flags)




