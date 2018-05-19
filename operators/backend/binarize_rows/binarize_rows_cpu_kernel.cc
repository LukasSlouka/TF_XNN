#include "tensorflow/core/framework/op.h"
#include "binarize_rows.h"

namespace tensorflow {
namespace functor {

    template <typename Dtype, typename Qtype>
    struct BinarizeRowsFunctor<CPUDevice, Dtype, Qtype> {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& input_matrix,
            Tensor* output_matrix,
            const int input_cols
        )
        {
            // Flatten the tensors
            const Dtype* input_flat = input_matrix.template flat<Dtype>().data();
            Qtype* output_flat = output_matrix->template flat<Qtype>().data();

            const int input_step = sizeof(Qtype) << 3;
            const int N = input_matrix.NumElements();

            int sign, compact;
            int bound = input_cols;
            int k = 0;

            // Iterate over over the input elements
            for (int i = 0; i < N; i += input_step) {
                compact = 0;
                for (int j = 0; (j < input_step) && (i + j < bound); j++) {
                    sign = input_flat[i + j] >= 0;
                    compact |= sign << (input_step - 1 - j);
                }
                output_flat[k] = compact;
                k++;
                if (i + input_step >= bound) {
                    i = bound - input_step;
                    bound += input_cols;
                }
            }
        }
    };

    template struct BinarizeRowsFunctor<CPUDevice, int8_t, int8_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int16_t, int8_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int32_t, int8_t>;
    template struct BinarizeRowsFunctor<CPUDevice, float, int8_t>;
    template struct BinarizeRowsFunctor<CPUDevice, double, int8_t>;

    template struct BinarizeRowsFunctor<CPUDevice, int8_t, int16_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int16_t, int16_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int32_t, int16_t>;
    template struct BinarizeRowsFunctor<CPUDevice, float, int16_t>;
    template struct BinarizeRowsFunctor<CPUDevice, double, int16_t>;

    template struct BinarizeRowsFunctor<CPUDevice, int8_t, int32_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int16_t, int32_t>;
    template struct BinarizeRowsFunctor<CPUDevice, int32_t, int32_t>;
    template struct BinarizeRowsFunctor<CPUDevice, float, int32_t>;
    template struct BinarizeRowsFunctor<CPUDevice, double, int32_t>;

} // namespace functor
} // namespace tensorflow
