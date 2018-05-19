#ifndef BINARIZE_ROWS_H_
#define BINARIZE_ROWS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

    class OpKernelContext;
    class Tensor;

    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;
}


namespace tensorflow {
namespace functor {

    template <typename Device, typename Dtype, typename Qtype>
    struct BinarizeRowsFunctor {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& input_matrix,
            Tensor* output_matrix,
            const int input_cols
        );
    };

} // functor
} // tensorflow

#endif // BINARIZE_ROWS_H_
