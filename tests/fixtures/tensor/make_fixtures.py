"""Generates the synthetic ONNX graphs used by the native tensor-inference tests.

These stand in for the real BS-RoFormer export (never committed; download-only)
so that the native OnnxTensorInference path demonstrably executes in CI.

    python -m pip install onnx==1.23.1
    python tests/fixtures/tensor/make_fixtures.py

Opset 17 / IR 8 are pinned so ONNX Runtime 1.30.0 loads every graph.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OPSET = [helper.make_opsetid("", 17)]
IR_VERSION = 8


def save(graph, name, **kwargs):
    model = helper.make_model(graph, opset_imports=OPSET, producer_name="automix-fixtures")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, str(HERE / name), **kwargs)


def identity_mask():
    # BS-RoFormer geometry (spec 5.4): folded-stereo spectrum in, one complex
    # mask out. The mask is 1 + 0i everywhere, so separation is the identity.
    #   input  [1, 801, 4100]  ->  output [1, 1, 2050, 801, 2]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 801, 4100])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2050, 801, 2])
    inits = [
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array([1, 801, 2050, 2], dtype=np.int64), "split_shape"),
        numpy_helper.from_array(np.array([1.0, 0.0], dtype=np.float32), "one_plus_zero_i"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "stem_axis"),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "zero"], ["zeroed"]),
        helper.make_node("Reshape", ["zeroed", "split_shape"], ["reim"]),
        helper.make_node("Add", ["reim", "one_plus_zero_i"], ["mask_tf"]),
        helper.make_node("Transpose", ["mask_tf"], ["mask_ft"], perm=[0, 2, 1, 3]),
        helper.make_node("Unsqueeze", ["mask_ft", "stem_axis"], ["output"]),
    ]
    save(helper.make_graph(nodes, "identity_mask", [x], [y], inits), "identity_mask.onnx")


def int64_io():
    # A quantized/tokenised export whose I/O is not float32: must be refused at load.
    x = helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, 4])
    y = helper.make_tensor_value_info("tokens_out", TensorProto.INT64, [1, 4])
    nodes = [helper.make_node("Identity", ["tokens"], ["tokens_out"])]
    save(helper.make_graph(nodes, "int64_io", [x], [y]), "int64_io.onnx")


def external_data():
    # fp32 exports keep weights in a `.onnx.data` sidecar.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), "w")
    nodes = [helper.make_node("Add", ["x", "w"], ["y"])]
    save(helper.make_graph(nodes, "external_data", [x], [y], [weight]),
         "external_data.onnx",
         save_as_external_data=True,
         all_tensors_to_one_file=True,
         location="external_data.onnx.data",
         size_threshold=0)


if __name__ == "__main__":
    identity_mask()
    int64_io()
    external_data()
