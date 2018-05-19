#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

/**
 * Binarizes input matrix row-wise reducing size linearly to qtype bit-width
 * @param input_matrix : input matrix
 * @param qtype: quantization type (8, 16, 32 bits)
 * @param output_matrix : binarized matrix
 */
REGISTER_OP("BinarizeRows")
    .Attr("dtype: {int8, int16, int32, float16, float32, float64}")
    .Input("input_matrix: dtype")
    .Attr("qtype: {int8, int16, int32}")
    .Output("output_matrix: qtype");

} /* tensorflow */