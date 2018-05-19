#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

/**
 * Binarizes input matrix column-wise reducing size 32 times at best (not templated)
 * Output matrix is transposed (same as transpose + binarize_rows)
 * @param input_matrix : input matrix
 * @paraam output_matrix : binarized  and trasponsed matrix
 */
REGISTER_OP("BinarizeCols")
    .Attr("dtype: {int8, int16, int32, float16, float32, float64}")
    .Input("input_matrix: dtype")
    .Output("output_matrix: int32");

} /* tensorflow */