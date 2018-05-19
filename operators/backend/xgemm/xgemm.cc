#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "xgemm.h"
#include "../binarize_rows/binarize_rows.h"
#include "../binarize_cols/binarize_cols.h"

namespace tensorflow {

    template<typename Device, typename Dtype, typename Otype>
    class XGEMMOp : public OpKernel {

        public:

            explicit XGEMMOp(OpKernelConstruction* context) : OpKernel(context) {}

            /* Compute method must be thread-safe */
            void Compute(OpKernelContext* context) override {

                // Get the input tensors
                const Tensor& A = context->input(0);
                const Tensor& B = context->input(1);

                // Check that inputs are matrices.
                OP_REQUIRES(context, TensorShapeUtils::IsMatrix(A.shape()), errors::InvalidArgument("input A is not a matrix"));
                OP_REQUIRES(context, TensorShapeUtils::IsMatrix(B.shape()), errors::InvalidArgument("input B is not a matrix"));

                // Get dimensions
                const int A_rows = A.dim_size(0);
                const int A_cols = A.dim_size(1);
                const int B_rows = B.dim_size(0);
                const int B_cols = B.dim_size(1);

                // Check that matrices can be multiplied
                OP_REQUIRES(
                    context, A_cols == B_rows,
                    errors::InvalidArgument("Matrices can not be multiplied: ", A_cols, " not equal to ", B_rows, ".")
                );

                // Binarize A
                int bit_padding = 31 - ((A_cols - 1) % 32);
                int A_s_cols  = (A_cols + bit_padding) >> 5;
                TensorShape A_bin_shape({A_rows, A_s_cols});
                Tensor A_bin;
                OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, A_bin_shape, &A_bin));
                // Squeezing
                ::tensorflow::functor::BinarizeRowsFunctor<Device, Dtype, int32_t> () (
                    context, A, &A_bin, A_cols
                );

                // Transpose and binarize B
                TensorShape B_bin_shape({B_cols, A_s_cols});
                Tensor B_bin;
                OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, B_bin_shape, &B_bin));
                // Squeezing
                ::tensorflow::functor::BinarizeColsFunctor<Device, Dtype> () (
                    context, B, &B_bin
                );

                // Calculate reduce amount based of bit padding and block padding
                int block_padding = 31 - ((A_s_cols - 1) % 32);
                const int reduce_amount = bit_padding + (block_padding << 5);

                // Prepare the output
                TensorShape out_shape({A_rows, B_cols});
                Tensor* C = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &C));

                // Finally perform the matrix multiplication
                ::tensorflow::functor::XGEMMFunctor<GPUDevice, Otype> () (
                    context, A_bin, B_bin, C, reduce_amount
                );
            }
    };

#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype, Otype) \
  REGISTER_KERNEL_BUILDER(Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("dtype").TypeConstraint<Otype>("otype"), \
  OPNAME(NAME)<GPUDevice, Dtype, Otype>); \

// Register operation
REGISTER(XGEMM, int8_t, int32_t);
REGISTER(XGEMM, int16_t, int32_t);
REGISTER(XGEMM, int32_t, int32_t);
REGISTER(XGEMM, float, int32_t);
REGISTER(XGEMM, double, int32_t);

REGISTER(XGEMM, int8_t, float);
REGISTER(XGEMM, int16_t, float);
REGISTER(XGEMM, int32_t, float);
REGISTER(XGEMM, float, float);
REGISTER(XGEMM, double, float);

REGISTER(XGEMM, int8_t, double);
REGISTER(XGEMM, int16_t, double);
REGISTER(XGEMM, int32_t, double);
REGISTER(XGEMM, float, double);
REGISTER(XGEMM, double, double);

} // namespace tensor
