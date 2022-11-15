# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""microTVM cares a lot about the convolution + bias + requantize + fused ReLU use case. There have
been some accuracy issues in the past, so this test steps through a model (MobileNetV1) layer by
layer and ensures there is 1-1 correspondance at each step. This test would run way faster if we ran
the model all at once, but then we wouldn't know which layers had issues.

Furthermore, this test uses some in-development optimizations for microTVM that aren't part of the
main pipeline.
"""

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

import tvm
import tvm.testing
from tvm import meta_schedule, relay
from tvm.testing.aot import AOTTestModel, run_and_check, AOTCompiledTestModel
from tvm.relay.backend import Executor, Runtime
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER
from tvm.contrib.download import download_testdata
from test_generalized_conv2d import change_ndarray_layout


MODEL_URL = "https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
SAMPLE_URL = (
    "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/elephant-299.jpg"
)


@pytest.fixture(scope="module")
def interpreter():
    """Returns a TFLite interpreter with the MLPerf Tiny visual wakewords model loaded, with an
    elephant image run through it, and with all intermediate layer outputs saved."""
    # Download the reference model
    rel_model_path = "model_microtvm_mobilenetv1.tflite"
    file = download_testdata(MODEL_URL, rel_model_path, overwrite=False)

    # Load it into TensorFlow and allocate memory
    interpreter = tf.lite.Interpreter(file, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Download an image. The neuron activations are strange if we use random data or ones,
    # so downloading an image is useful.
    rel_image_path = "image_microtvm_mobilenetv1.jpg"
    img_path = download_testdata(SAMPLE_URL, rel_image_path, overwrite=False)
    image = Image.open(img_path).resize((96, 96))
    image_data_hwc_uint8 = np.asarray(image)
    assert image_data_hwc_uint8.shape == (96, 96, 3)
    assert image_data_hwc_uint8.dtype == "uint8"
    image_data_nhwc_int8 = (image_data_hwc_uint8 + 128).view("int8").reshape((1, 96, 96, 3))

    # Load the image into the TFLite interpreter and compute all intermediate tensor values
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], image_data_nhwc_int8)
    interpreter.invoke()
    return interpreter


def _get_layer_attributes(layer_num):
    """Returns the relevant padding and stride for a given layer in a MobileNetV1 model. It's a huge
    headache to read this data from TensorFlow, as it is not user accessible via the interpreter. If
    we really wanted to, we would have to parse the .tflite file ourselves. This function is a bit
    of a hack, but lets us skip that."""

    if layer_num == 0:  # Regular conv2d
        return ((0, 0, 1, 1), (2, 2))
    if layer_num % 2 == 0:  # 1x1 conv2d
        return ((0, 0, 0, 0), (1, 1))
    if layer_num in [3, 7, 11, 23]:  # Downsizing depthwise_conv2d layers
        return ((0, 0, 1, 1), (2, 2))
    # Depthwise conv2d
    return ((1, 1, 1, 1), (1, 1))


def _get_relu_activation_prefix(layer_num):
    if layer_num == 0:
        return "model/activation/Relu;"
    return f"model/activation_{layer_num}/Relu;"


def _get_main_path_tensor_details(details, tensor_num):
    """A "main path" tensor is a fused layer input/output. Gets the tensor details from the tensor
    index, where 0 gives the original input tensor, 1 gives the output of the first fused
    convolution layer, and so on. TFLite names are a little wack, so we get this information by
    finding the SECOND tensor (which has the suffix "1") for each ReLU activation (the first tensor
    is the bias)."""

    if tensor_num == 0:
        return details[0]
    prefix = _get_relu_activation_prefix(tensor_num - 1)
    detail = next(d for d in details if d["name"].startswith(prefix) and d["name"].endswith("1"))
    assert len(detail["shape"]) == 4
    assert detail["dtype"] == np.int8
    return detail


def _get_bias_details(details, layer_num):
    """Gets the tensor details for the bias tensor for the corresponding convolution layer. The
    bias tensors always appear before the main path tensors, so we don't have to check the ending to
    make sure we have the right one."""
    prefix = _get_relu_activation_prefix(layer_num)
    detail = next(d for d in details if d["name"].startswith(prefix))
    assert len(detail["shape"]) == 1
    assert detail["dtype"] == np.int32
    return detail


def _get_kernel_details(details, layer_num):
    """Gets the tensor details for the kernel tensor for the corresponding convolution layer. These
    have a different naming scheme from the main path and bias tensors, as they are converted before
    activation function fusion. Note that regular vs depthwise conv2ds have different prefixes."""

    if layer_num == 0:
        prefix = "model/conv2d/Conv2D"
    elif layer_num % 2 == 0:
        prefix = f"model/conv2d_{layer_num // 2}/"
    else:
        prefix = f"model/batch_normalization_{layer_num}/"

    detail = next(d for d in details if d["name"].startswith(prefix))
    assert len(detail["shape"]) == 4
    assert detail["dtype"] == np.int8
    return detail


def _get_quant_scale_const(quantization_dict, as_scalar=False):
    scales = quantization_dict["scales"]
    if as_scalar:
        assert len(scales) == 1
        scales = scales[0]
    return relay.const(scales, "float32")


def _get_quant_zp_const(quantization_dict, as_scalar=False):
    zero_points = quantization_dict["zero_points"]
    if as_scalar:
        assert len(zero_points) == 1
        zero_points = zero_points[0]
    return relay.const(zero_points, "int32")


def _change_layout(data, old_layout, new_layout, dtype):
    return change_ndarray_layout(data, old_layout, new_layout).astype(dtype)


def _load_tflite_layer(interpreter, layer):
    tensor_details = interpreter.get_tensor_details()

    def lookup(detail):
        return interpreter.get_tensor(detail["index"]), detail["quantization_parameters"]

    input_data = lookup(_get_main_path_tensor_details(tensor_details, layer))
    kernel_data = lookup(_get_kernel_details(tensor_details, layer))
    bias_data = lookup(_get_bias_details(tensor_details, layer))
    output_data = lookup(_get_main_path_tensor_details(tensor_details, layer + 1))

    return input_data, kernel_data, bias_data, output_data


def _make_relay_partial_func(relay_op, *args, **kwargs):
    return lambda op: relay_op(op, *args, **kwargs)


def _make_conv2d_op(kernel, data_quant, kernel_quant, hyperparams, is_depthwise=False):
    dtype, padding, strides, data_layout, kernel_layout, output_layout = hyperparams
    kernel_size = kernel.shape[1:3]
    if is_depthwise:
        channels = groups = kernel.shape[3]
    else:
        channels = kernel.shape[0]
        groups = 1

    kernel_ndarr = _change_layout(kernel, "OHWI", kernel_layout, dtype)

    return _make_relay_partial_func(
        relay.qnn.op.conv2d,
        relay.const(kernel_ndarr, dtype),
        input_zero_point=_get_quant_zp_const(data_quant, as_scalar=True),
        kernel_zero_point=_get_quant_zp_const(kernel_quant),
        input_scale=_get_quant_scale_const(data_quant, as_scalar=True),
        kernel_scale=_get_quant_scale_const(kernel_quant),
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        dilation=(1, 1),
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype="int32",
        out_layout=output_layout,
    )


def _make_bias_op(bias, output_layout):
    requantize_axis = output_layout.index("C")
    return _make_relay_partial_func(
        relay.op.nn.bias_add,
        relay.const(bias, "int32"),
        axis=requantize_axis,
    )


def _make_requantize_op(bias_quant, output_quant, output_dtype, output_layout):
    requantize_axis = output_layout.index("C")
    return _make_relay_partial_func(
        relay.qnn.op.requantize,
        _get_quant_scale_const(bias_quant),
        _get_quant_zp_const(bias_quant),
        _get_quant_scale_const(output_quant, as_scalar=True),
        _get_quant_zp_const(output_quant, as_scalar=True),
        axis=requantize_axis,
        compute_dtype="int64",
        out_dtype=output_dtype,
    )


def _make_aot_model(params, hyperparams, layouts, is_depthwise=False):
    tensors, quantizations = zip(*params)
    data, kernel, bias, output = tensors
    data_quant, kernel_quant, bias_quant, output_quant = quantizations

    dtype, padding, _strides = hyperparams
    data_layout, _, output_layout = layouts

    if any(padding):
        pad_const = int(data_quant["zero_points"][0])
        pad_before = (0, padding[0], padding[1], 0)
        pad_after = (0, padding[2], padding[3], 0)
        data = np.pad(data, tuple(zip(pad_before, pad_after)), constant_values=pad_const)
    data_ndarr = _change_layout(data, "NHWC", data_layout, dtype)

    output_ndarr = _change_layout(output, "NHWC", output_layout, dtype)

    input_var = relay.var("input", relay.TensorType(data_ndarr.shape, dtype))
    conv2d = _make_conv2d_op(kernel, data_quant, kernel_quant, hyperparams + layouts, is_depthwise)
    bias = _make_bias_op(bias, output_layout)
    requantize = _make_requantize_op(bias_quant, output_quant, dtype, output_layout)

    relay_mod = requantize(bias(conv2d(input_var)))
    relay_func = relay.Function([input_var], relay_mod)
    return AOTTestModel(
        module=tvm.IRModule.from_expr(relay_func),
        inputs={"input": data_ndarr},
        outputs={"output": output_ndarr},
        output_tolerance=1,
    )


def _make_target():
    return tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m7")


def _make_executor():
    return Executor(
        "aot",
        {
            "workspace-byte-alignment": 8,
            "constant-byte-alignment": 8,
            "interface-api": "c",
            "unpacked-api": True,
        },
    )


@pytest.mark.parametrize("layer", range(0, 23, 2))
@tvm.testing.requires_corstone300
def test_qnn_conv2d_mobilenetv1_layer(interpreter, layer):
    dtype = "int16"

    tensor, kernel, bias, output = _load_tflite_layer(interpreter, layer)
    padding, strides = _get_layer_attributes(layer)

    if layer % 2 == 0:
        data_layout, kernel_layout, output_layout = "NHWC", "OHWI", "NHWC"
    else:
        data_layout, kernel_layout, output_layout = "NCHW", "OIHW", "NCHW"

    test_model = _make_aot_model(
        (tensor, kernel, bias, output),
        (dtype, padding, strides),
        (data_layout, kernel_layout, output_layout),
        is_depthwise=layer % 2 == 1,
    )

    def schedule_fn(_sch):
        return True

    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "allow_extern",
        },
        disabled_pass=["qnn.Legalize"],
    ), meta_schedule.database.ScheduleFnDatabase(schedule_fn):
        executor_factory = tvm.relay.build(
            test_model.module,
            _make_target(),
            executor=_make_executor(),
            runtime=Runtime("crt"),
            params=test_model.params,
            mod_name=test_model.name,
        )
        compiled = AOTCompiledTestModel(model=test_model, executor_factory=executor_factory)

    run_and_check(
        models=[compiled],
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        workspace_byte_alignment=8,
        constant_byte_alignment=8,
    )
