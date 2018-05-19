#ifndef XGEMM_H_
#define XGEMM_H_

#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {

    class OpKernelContext;
    class Tensor;

    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;
}


namespace tensorflow {
namespace functor {

    template <typename Device, typename Otype>
    struct XGEMMFunctor {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& matrix_a,
            const Tensor& matrix_b,
            Tensor* output_matrix,
            const int reduce_amount
        );
    };

} // functor
} // tensorflow

#endif // XGEMM_H_
