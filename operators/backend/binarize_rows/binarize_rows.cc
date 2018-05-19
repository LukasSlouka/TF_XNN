#include "binarize_rows.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

    template<typename Device, typename Dtype, typename Qtype>
    class BinarizeRowsOp : public OpKernel {

        public:

            explicit BinarizeRowsOp(OpKernelConstruction* context) : OpKernel(context) {}

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
                int out_element_size = sizeof(Qtype) << 3;

                // Prepare the output
                int output_cols  = (cols + out_element_size - 1) / out_element_size;

                TensorShape out_shape({rows, output_cols});
                Tensor* output_matrix = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_matrix));

                // Finally perform binarization
                ::tensorflow::functor::BinarizeRowsFunctor<Device, Dtype, Qtype> () (
                    context,
                    input_matrix,
                    output_matrix,
                    cols
                );
            }
    };

#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype, Qtype) \
  REGISTER_KERNEL_BUILDER( \
      Name(#NAME).Device(DEVICE_CPU).TypeConstraint<Dtype>("dtype").TypeConstraint<Qtype>("qtype"), \
      OPNAME(NAME)<CPUDevice, Dtype, Qtype>); \
  REGISTER_KERNEL_BUILDER( \
      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("dtype").TypeConstraint<Qtype>("qtype"), \
      OPNAME(NAME)<GPUDevice, Dtype, Qtype>); \


// Register operation
REGISTER(BinarizeRows, int8_t, int8_t);
REGISTER(BinarizeRows, int16_t, int8_t);
REGISTER(BinarizeRows, int32_t, int8_t);
REGISTER(BinarizeRows, float, int8_t);
REGISTER(BinarizeRows, double, int8_t);

REGISTER(BinarizeRows, int8_t, int16_t);
REGISTER(BinarizeRows, int16_t, int16_t);
REGISTER(BinarizeRows, int32_t, int16_t);
REGISTER(BinarizeRows, float, int16_t);
REGISTER(BinarizeRows, double, int16_t);

REGISTER(BinarizeRows, int8_t, int32_t);
REGISTER(BinarizeRows, int16_t, int32_t);
REGISTER(BinarizeRows, int32_t, int32_t);
REGISTER(BinarizeRows, float, int32_t);
REGISTER(BinarizeRows, double, int32_t);

} // namespace tensor
