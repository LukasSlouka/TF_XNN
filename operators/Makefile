# TensorFlow includes and libraries
TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# !! Change these to fit your system !!!
TF_NSYNC=/home/lslouka/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public
TF_CUDA_CONFIG=/home/lslouka/tensorflow/lib/python3.6/site-packages/external/local_config_cuda/cuda

# Compilation flags
FLAGS=-fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=1

# Directories for backend, frontend and object files
BE=backend
FE=frontend
OBJ=obj

# CUDA targets
CUDA_TARGETS=$(OBJ)/binarize_rows.cu.o $(OBJ)/xgemm.cu.o $(OBJ)/binarize_cols.cu.o


all: build

build: operators.so ${CUDA_TARGETS}

# CUDA targets
$(OBJ)/binarize_rows.cu.o: $(OBJ)
	nvcc -std=c++11 --expt-relaxed-constexpr --shared -Xcompiler -O3 \
	  -c -o $(OBJ)/binarize_rows.cu.o $(BE)/binarize_rows/binarize_rows_gpu_kernel.cu.cc \
	  -isystem $(TF_INC) \
	  -isystem $(TF_NSYNC) \
	  -isystem $(TF_CUDA_CONFIG) \
	  -L$(TF_LIB) -ltensorflow_framework \
	  -D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)

$(OBJ)/binarize_cols.cu.o: $(OBJ)
	nvcc -std=c++11 --expt-relaxed-constexpr --shared -O3 \
	  -c -o $(OBJ)/binarize_cols.cu.o $(BE)/binarize_cols/binarize_cols_gpu_kernel.cu.cc \
	  -isystem $(TF_INC) \
	  -isystem $(TF_NSYNC) \
	  -isystem $(TF_CUDA_CONFIG) \
	  -L$(TF_LIB) -ltensorflow_framework \
	  -D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)

$(OBJ)/xgemm.cu.o: $(OBJ)
	nvcc -std=c++11 --expt-relaxed-constexpr --shared -Xcompiler -O3 \
	  -c -o $(OBJ)/xgemm.cu.o $(BE)/xgemm/xgemm_gpu_kernel.cu.cc \
	  -isystem $(TF_INC) \
	  -isystem $(TF_NSYNC) \
	  -isystem $(TF_CUDA_CONFIG) \
	  -L$(TF_LIB) -ltensorflow_framework \
	  -D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)

# Shared library
operators.so: ${CUDA_TARGETS} $(BE)/binarize_rows/binarize_rows.cc $(BE)/binarize_cols/binarize_cols.cc $(BE)/xgemm/xgemm.cc
	g++ -std=c++11 -O3 -o operators.so --shared \
	  $(BE)/binarize_rows/binarize_rows.cc $(FE)/binarize_rows.cc $(BE)/binarize_rows/binarize_rows_cpu_kernel.cc $(OBJ)/binarize_rows.cu.o \
	  $(BE)/xgemm/xgemm.cc $(FE)/xgemm.cc $(OBJ)/xgemm.cu.o \
	  $(BE)/binarize_cols/binarize_cols.cc $(FE)/binarize_cols.cc $(OBJ)/binarize_cols.cu.o \
	  -isystem $(TF_INC)  \
	  -isystem $(TF_NSYNC) \
	  -isystem $(TF_CUDA_CONFIG) \
	  -lcudart \
	  -L$(TF_LIB) -ltensorflow_framework \
	  $(FLAGS)

clean:
	rm -f ${CUDA_TARGETS} *.so *.pyc *.npy *.o

$(OBJ):
	mkdir $@
