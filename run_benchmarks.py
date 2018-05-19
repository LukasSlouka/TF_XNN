from benchmarks.gemm_benchmark import gemm_benchmark

if __name__ == '__main__':
    for batch_size in range(100, 4001, 100):
        gemm_benchmark(run_tf=True, out_path='logs/bm_xgemm_2', batch_size=batch_size, min=100, max=4001, step=100, iterations=10)
