#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


namespace tensorflow {

/**
 * XNOR general matrix multiply operator (XGEMM)
 * @param dtype : type of input matrices (matrices are expected to be quantized)
 * @param matrix_a : first input matrix
 * @param matrix_b : second input matrix
 * @param matrix_c : output of matrix multiplication
 */
REGISTER_OP("XGEMM")
    .Attr("dtype: {int8, int16, int32, float16, float32, float64}")
    .Attr("otype: {int32, float32, float64}")
    .Input("matrix_a: dtype")
    .Input("matrix_b: dtype")
    .Output("matrix_c: otype")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

} /* tensorflow */
