#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define BLOCK_SIZE 32  // dont change unless you dont want 32-bit integer squeeze

#include "binarize_cols.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace {

    template <typename Dtype>
    __global__ void binarize_cols_kernel(
        const int output_rows,
        const int output_cols,
        const int input_rows,
        const int input_cols,
        const int last_max,
        const Dtype *flat_input,
        int32_t *flat_output
    )
    {
        // Set shared memory
        __shared__ Dtype mem[BLOCK_SIZE][BLOCK_SIZE];

        // Get row and column
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        // initialize shared mem to -1
        for (int i = 0; i < BLOCK_SIZE; i++)
            mem[threadIdx.y][i] = -1;

        __syncthreads();

        int read_in_col, read_in_row = (col << 5) + threadIdx.y;
        if (read_in_row < input_rows) {
            read_in_col = blockIdx.y << 5;
            // load shared memory continuously
            if (read_in_col + BLOCK_SIZE >= input_cols)
                for (int j = 0; j < last_max; j += 1)
                    mem[threadIdx.y][j] = flat_input[read_in_row * input_cols + read_in_col + j];
            else
                for (int j = 0; j < BLOCK_SIZE; j += 1)
                    mem[threadIdx.y][j] = flat_input[read_in_row * input_cols + read_in_col + j];
        }

        __syncthreads();

        // each thread squeezes their column of shared memory and saves to output
        if (row < output_rows) {
            int sign, compact = 0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                sign = mem[i][threadIdx.y] >= 0;
                compact |= sign << (BLOCK_SIZE - 1 - i);
            }
            flat_output[row * output_cols + col] = compact;
        }
    }
}

namespace tensorflow {
namespace functor {

    template <typename Dtype>
    struct BinarizeColsFunctor<GPUDevice, Dtype> {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& input_matrix,
            Tensor *output_matrix
        )
        {
            int input_cols = input_matrix.dim_size(1);
            int input_rows = input_matrix.dim_size(0);
            int last_max = input_cols % BLOCK_SIZE;
            if (last_max == 0)
                last_max = BLOCK_SIZE;
            const int output_rows = output_matrix->dim_size(0);
            const int output_cols = output_matrix->dim_size(1);

            auto input_flat = input_matrix.template flat<Dtype>().data();
            auto output_flat = output_matrix->template flat<int32_t>().data();

            // 1 column
            dim3 dimBlock(1, BLOCK_SIZE, 1);

            dim3 dimGrid(
                output_cols,
                (output_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            binarize_cols_kernel<Dtype>
                <<<dimGrid, dimBlock>>>(
                    output_rows,
                    output_cols,
                    input_rows,
                    input_cols,
                    last_max,
                    input_flat,
                    output_flat
                );
        }
    };

    template struct BinarizeColsFunctor<GPUDevice, int8_t>;
    template struct BinarizeColsFunctor<GPUDevice, int16_t>;
    template struct BinarizeColsFunctor<GPUDevice, int32_t>;
    template struct BinarizeColsFunctor<GPUDevice, float>;
    template struct BinarizeColsFunctor<GPUDevice, double>;

} // namespace functor
} // namespace tensorflow

#endif // GOOGLE_CUDA
