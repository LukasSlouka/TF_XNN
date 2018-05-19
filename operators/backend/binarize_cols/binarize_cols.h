#ifndef BINARIZE_COLS_H_
#define BINARIZE_COLS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

    class OpKernelContext;
    class Tensor;

    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;
}


namespace tensorflow {
namespace functor {

    template <typename Device, typename Dtype>
    struct BinarizeColsFunctor {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& input_matrix,
            Tensor* output_matrix
        );
    };

} // functor
} // tensorflow

#endif // BINARIZE_COLS_H_
