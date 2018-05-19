#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "binarize_rows.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace {

    template <typename Dtype>
    __device__ int binarize(
        const Dtype* a,
        const int count,
        const int item_size)
    {
        int sign, compact = 0;
        for (int i = 0; i < count; i++) {
            sign = a[i] >= 0;
            compact |= sign << (item_size - 1 - i);
        }
        return compact;
    }

    template <typename Dtype, typename Qtype>
    __global__ void binarize_rows_kernel(
        const int output_rows,
        const int output_cols,
        const int input_cols,
        const int item_size,
        const int last_max,
        const Dtype *flat_input,
        Qtype *flat_output
    )
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if ((row < output_rows) && (col < output_cols)) {
            int in_idx = row * input_cols + col * item_size;
            if (col == output_cols - 1)
                flat_output[row * output_cols + col] = binarize<Dtype>(&flat_input[in_idx], last_max, item_size);
            else
                flat_output[row * output_cols + col] = binarize<Dtype>(&flat_input[in_idx], item_size, item_size);
        }
    }
}

namespace tensorflow {
namespace functor {

    template <typename Dtype, typename Qtype>
    struct BinarizeRowsFunctor<GPUDevice, Dtype, Qtype> {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& input_matrix,
            Tensor *output_matrix,
            const int input_cols
        )
        {
            // CUDA block set to 32
            const int BLOCK_SIZE = 32;

            const int item_size = sizeof(Qtype) << 3;
            int last_max = input_cols % item_size;
            if (last_max == 0)
                last_max = item_size;
            const int output_rows = output_matrix->dim_size(0);
            const int output_cols = output_matrix->dim_size(1);

            auto input_flat = input_matrix.template flat<Dtype>();
            auto output_flat = output_matrix->template flat<Qtype>();

            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid(
                (output_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            binarize_rows_kernel<Dtype, Qtype>
                <<<dimBlock, dimGrid>>>(
                    output_rows, output_cols, input_cols, item_size, last_max,
                    input_flat.data(),
                    output_flat.data()
                );
        }
    };

    template struct BinarizeRowsFunctor<GPUDevice, int8_t, int8_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int16_t, int8_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int32_t, int8_t>;
    template struct BinarizeRowsFunctor<GPUDevice, float, int8_t>;
    template struct BinarizeRowsFunctor<GPUDevice, double, int8_t>;

    template struct BinarizeRowsFunctor<GPUDevice, int8_t, int16_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int16_t, int16_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int32_t, int16_t>;
    template struct BinarizeRowsFunctor<GPUDevice, float, int16_t>;
    template struct BinarizeRowsFunctor<GPUDevice, double, int16_t>;

    template struct BinarizeRowsFunctor<GPUDevice, int8_t, int32_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int16_t, int32_t>;
    template struct BinarizeRowsFunctor<GPUDevice, int32_t, int32_t>;
    template struct BinarizeRowsFunctor<GPUDevice, float, int32_t>;
    template struct BinarizeRowsFunctor<GPUDevice, double, int32_t>;

} // namespace functor
} // namespace tensorflow

#endif // GOOGLE_CUDA
