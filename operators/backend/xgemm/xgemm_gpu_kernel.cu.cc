#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "xgemm.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define BLOCK_SIZE 32
#define BLOCK_SHIFT 5

namespace {

    template <typename Otype>
    __global__ void xgemm_kernel(
        const int32_t *A,
        const int32_t *B,
        Otype *C,
        const int32_t A_rows, // C_rows
        const int32_t A_cols, // B_rows
        const int32_t B_cols, // C_cols
        const int reduce_amount
    )
    {
        // accumulator
        Otype c_value = 0;

        // row in the output matrix (row of A)
        int32_t row = (blockIdx.y << BLOCK_SHIFT) + threadIdx.y;

        // column in the output matrix (column of B)
        int32_t col = (blockIdx.x << BLOCK_SHIFT) + threadIdx.x;

        // Helper variables
        int32_t block_start;
        int32_t shm_idx = (threadIdx.y  << BLOCK_SHIFT) + threadIdx.x;
        int32_t blocks = (A_cols + BLOCK_SIZE - 1) >> BLOCK_SHIFT;

        // Prepare shared memory (submatrices of A and B)
        __shared__ int32_t As[BLOCK_SIZE * (BLOCK_SIZE + 1)];
        __shared__ int32_t Bs[BLOCK_SIZE * (BLOCK_SIZE + 1)];

        // k iterates over blocks row wise (for A, column wise for B)
        for (unsigned int k = 0; k < blocks; k++) {

            block_start = k << BLOCK_SHIFT;

            // each thread fills one element in As by value from A or by 0
            if (block_start + threadIdx.x < A_cols && row < A_rows)
                As[shm_idx] = A[row * A_cols + block_start + threadIdx.x];
            else
                As[shm_idx] = 0;

            // memory coalescing, B is transposed, so we can access it row wise
            if (block_start + threadIdx.y < A_cols && col < B_cols)
//                Bs[shm_idx] = B[(block_start + threadIdx.y) * B_cols + col];
                Bs[shm_idx] = B[col * A_cols + block_start + threadIdx.y];   // coalesced access (B is transposed)
            else
                Bs[shm_idx] = 0;

            // sync up
            __syncthreads();

            // compute c value
            for (unsigned int n = 0; n < BLOCK_SIZE; ++n)
                // c_value += (__popc( ~(As[threadIdx.y * BLOCK_SIZE + n] ^ Bs[n * BLOCK_SIZE + threadIdx.x]) ) << 1 ) - 32;  // XNOR operation
                c_value += __popc(As[(threadIdx.y << BLOCK_SHIFT) + n] ^ Bs[(n << BLOCK_SHIFT) + threadIdx.x]);  // popcnt + XOR

            // sync
            __syncthreads();
        }

        // store computed result if it is within the bounds
        if (row < A_rows && col < B_cols)
            // c value contains counts of 0 (we wanted counts of 1)
            C[row * B_cols + col] = blocks * (BLOCK_SIZE << 5) - (c_value * 2) - reduce_amount;
    }
}


namespace tensorflow {
namespace functor {

    template <typename Otype>
    struct XGEMMFunctor<GPUDevice, Otype> {

        void operator() (
            ::tensorflow::OpKernelContext *context,
            const Tensor& matrix_a,
            const Tensor& matrix_b,
            Tensor* output_matrix,
            const int reduce_amount
        )
        {
            // prepare inputs
            const int A_rows = matrix_a.dim_size(0);
            const int A_cols = matrix_a.dim_size(1);
            // const int B_cols = matrix_b.dim_size(1);
            const int B_cols = matrix_b.dim_size(0); // B is transposed

            auto A = matrix_a.template flat<int32_t>().data();
            auto B = matrix_b.template flat<int32_t>().data();
            auto C = output_matrix->template flat<Otype>().data();

            // Set block and grid dimensions
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid(
                (B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            // Aaaand compute
            xgemm_kernel<Otype>
            <<<dimGrid, dimBlock>>>(
                A, B, C, A_rows, A_cols, B_cols, reduce_amount
            );
        }
    };

    template struct XGEMMFunctor<GPUDevice, int32_t>;
    template struct XGEMMFunctor<GPUDevice, float>;
    template struct XGEMMFunctor<GPUDevice, double>;

} // namespace functor
} // namespace tensorflow

#endif // GOOGLE_CUDA
