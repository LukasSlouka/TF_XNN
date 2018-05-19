#include "binarize_cols.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {

    template<typename Device, typename Dtype>
    class BinarizeColsOp : public OpKernel {

        public:

            explicit BinarizeColsOp(OpKernelConstruction* context) : OpKernel(context) {}

            /* Compute method must be thread-safe */
            void Compute(OpKernelContext* context) override {

                // Get the input matrix
                const Tensor& input_matrix = context->input(0);

                // Works only on 2D matrix
                OP_REQUIRES(
                    context, TensorShapeUtils::IsMatrix(input_matrix.shape()), errors::InvalidArgument("In[0] is not a matrix")
                );

                // Get dimensions
                const int rows = input_matrix.dim_size(0);
                const int cols = input_matrix.dim_size(1);

                // Prepare the output (binarized transposed)
                int output_cols  = (rows + 31) / 32;
                TensorShape out_shape({cols, output_cols});
                Tensor* output_matrix = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_matrix));

                // Finally perform binarization
                ::tensorflow::functor::BinarizeColsFunctor<Device, Dtype> () (
                    context,
                    input_matrix,
                    output_matrix
                );
            }
    };

#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype) \
  REGISTER_KERNEL_BUILDER(Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("dtype"), OPNAME(NAME)<GPUDevice, Dtype>); \


// Register operation
REGISTER(BinarizeCols, int8_t);
REGISTER(BinarizeCols, int16_t);
REGISTER(BinarizeCols, int32_t);
REGISTER(BinarizeCols, float);
REGISTER(BinarizeCols, double);

} // namespace tensor
