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
"""Arm Cortex-M specific optimizations for quantized operators."""

import numpy as np

from tvm import nd, relay, target
from ..nn import *

import pdb

def _prev_ops_match(curr_op, pattern):
    prev_op = curr_op
    for op_name in pattern:
        print(f"Looking for {op_name}")
        prev_op = prev_op.args[0]
        if (not hasattr(prev_op, "op")) or prev_op.op.name != op_name:
            print(f"Broke on {prev_op.op.name}")
            return False
    return True


def _edit_attrs(attrs, **kwargs):
    print(attrs)
    return {**attrs, **kwargs}


def change_numpy_layout(arr, src_layout, dst_layout):
    assert src_layout.isalpha() and dst_layout.isalpha()
    axis_order = [src_layout.index(c) for c in dst_layout]
    return np.transpose(arr, axis_order)


def _squash_transformations(kernel_op):
    if isinstance(kernel_op, relay.expr.Constant):
        return kernel_op.data.numpy()
    assert isinstance(kernel_op, relay.expr.Call)
    assert len(kernel_op.args) == 1

    prev_kernel = _squash_kernel_transformations(kernel_op.args[0])
    attrs = kernel_op.attrs

    if kernel_op.op.name == "layout_transform":
        return change_numpy_layout(prev_kernel, attrs.src_layout, attrs.dst_layout)
    elif kernel_op.op.name == "cast":
        return prev_kernel.astype(attrs.dtype)
    elif kernel.op.name == "expand_dims":
        new_axes = range(attrs.axis, attrs.axis + attrs.num_newaxis)
        return np.expand_dims(prev_kernel, tuple(axes))
    else:
        raise RuntimeError(f"Invalid kernel transformation '{kernel_op}'!")


def _remove_empty_channels(source_array, channel_dim, factor=2):
    """
    """
    assert channel_dim < 2
    num_channels = source_array.shape[channel_dim]
    non_empty_channels = []
    indices = []

    for i in range(num_channels):
        channel = array.take(indices=i, axis=channel_dim)
        if np.any(channel):
            non_empty_channels.append(channel)
            indices.append(i)


    '''lowest_empty_channel = 0
    while indices % factor != 0:
        # Find the lowest index that is empty
        while lowest_empty_channel in indices:
            lowest_empty_channel += 1


        lowest_empty_channel = min(set(range(num_channels)) - indices)
        non_empty_channels.append(lowest_empty_channel)
        indices.add(lowest_empty_channel)'''

    print(f"Slicing away empty channels kept {len(non_empty_channels)}/{num_channels} channels!")
    return np.stack(non_empty_channels, axis=channel_dim), indices


def _compute_fixed_conv2d_outputs(conv2d_op, add_op, requantize_op):
    """Compute all conv2d output values that do not depend on the layer input."""

    assert conv2d_op.attrs.kernel_layout == "OHWI"
    assert conv2d_op.attrs.groups == 1
    kernel = _squash_transformations(conv2d_op.args[1])

    num_channels = kernel.shape[3]
    rq_input_scale = requantize_op.args[1].data.numpy()
    rq_output_scale = requantize_op.args[3].data.numpy().item()
    bias_data = _squash_transformations(add_op.args[1]).flatten()

    outputs = {}

    for i in range(num_channels):
        if np.any(kernel[i, :, :, :]):
            continue
        scale = rq_input_scale[i] / rq_output_scale
        channel_constant = round(bias_data[i] * scale + out_zero_point)
        clipped = min(127, max(-128, channel_constant))
        outputs[i] = clipped

    return kernel, outputs


def alter_depthwise_conv2d_layout(depthwise_conv2d):
    cast_op = depthwise_conv2d.args[0]
    requantize_op = cast_op.args[0]
    add_op = requantize_op.args[0]
    prev_conv2d_op = add_op.args[0]

    return relay.qnn.op.conv2d(
        relay.layout_transform(
            relay.cast(
                relay.qnn.op.requantize(
                    relay.op.add(
                        relay.qnn.op.conv2d(
                            *prev_conv2d_op.args,
                            **_edit_attrs(prev_conv2d_op.attrs, out_layout="NCHW"),
                        ),
                        relay.layout_transform(
                            add_op.args[1],
                            src_layout="NHWC",
                            dst_layout="NCHW",
                        )
                    ),
                    *requantize_op.args[1:],
                    **_edit_attrs(requantize_op.attrs, axis=1),
                ),
                dtype="int16",
            ),
            src_layout="NCHW",
            dst_layout="NHWC",
        ),
        *depthwise_conv2d.args[1:],
        **_edit_attrs(depthwise_conv2d.attrs, data_layout="NCHW"),
    )


def strip_zero_conv2d_channels(regular_conv2d):
    """Replaces a conv2d op with all zero channels with an equivalent dense operator.

    We must operate on (regular -> depthwise -> regular) sequences of conv2d blocks.
    """


@qnn_conv2d_alter_layout.register(["arm_cpu"])
def alter_conv2d_layout(attrs, inputs, _tinfos, _out_type):
    """Adjust a qnn.conv2d and preceeding ops to better fit on Cortex-M.
    """
    current_target = target.Target.current(allow_none=False)
    if not "cortex-m" in current_target.mcpu:
        return None

    # Always cast to int16 and pick a our desired kernel layout - this won't affect anything
    data_expr, kernel_expr = inputs[:2]
    is_depthwise = attrs.groups > 1
    new_kernel_layout = "IOHW" if is_depthwise else "OHWI"

    op = relay.qnn.op.conv2d(
        relay.cast(data_expr, dtype="int16"),
        relay.cast(kernel_expr, dtype="int16"),
        *inputs[2:],
        **_edit_attrs(
            attrs,
            kernel_layout=new_kernel_layout,
            out_layout="NHWC"
        ),
    )

    # If possible, modify depthwise ops to take as input NCHW instead.
    if is_depthwise and _prev_ops_match(op, ("cast", "qnn.requantize", "add", "qnn.conv2d")):
        op = alter_depthwise_conv2d_layout(op)

    return op


@qnn_add_alter_layout.register(["arm_cpu"])
def alter_add_layout(_attrs, inputs, _tinfos, _out_type):
    """Fuses the zero point for a previous quantized operator with this add operation.

    Currently only supports qnn.conv2d, but qnn.dense support should be added. Note that this
    optimization means we must pad tensors with the input zero point, and NOT with zero.
    """
    return None
    prev_op, biases_data_op = inputs
    if not hasattr(prev_op, "op"):
        return None
    if prev_op.op.name != "qnn.conv2d":
        return None

    # We should not perform this alteration if the target has a uint * int SIMD MAC operation (since
    # these do (x - (-128)) * y efficiently, and conv_input_zp is usually -128). For now, we
    # restrict this optimization to just Cortex-M devices, but it might be helpful on others too.
    current_target = target.Target.current(allow_none=False)
    if not "cortex-m" in current_target.mcpu:
        return None

    conv_input_zp = prev_op.args[2].data.numpy().item()
    kernel = prev_op.args[1].data.numpy()

    if _is_qnn_op_depthwise_conv2d(prev_op.attrs, prev_op.args):
        axes_to_sum = "HW"
    elif prev_op.attrs.groups == 1:
        axes_to_sum = "HWI"
    else:
        # This alteration does not currently support grouped conv2d
        return None
    axes_to_sum = tuple(map(prev_op.attrs.kernel_layout.index, axes_to_sum))
    element_sums = np.sum(kernel, axis=axes_to_sum).flatten()

    # The zero point is subtracted from the input elements, so we need a "-" sign here
    zp_shifted_sums = element_sums * (-conv_input_zp)

    # The bias values may or may not be wrapped in an expand_dims op
    if isinstance(biases_data_op, relay.expr.Call):
        biases = biases_data_op.args[0]
    else:
        biases = biases_data_op
    assert isinstance(biases, relay.expr.Constant)

    # We want to make sure new_biases is representable as an int32. It's tempting to just check
    # whether arr.dtype == "int32" (since Numpy will automatically increase dtype in some cases)
    # but this leads to weird wrapping behavior and doesn't work. We must do it manually.
    new_biases = biases.data.numpy().astype("int64") + zp_shifted_sums
    if new_biases.min() < -(2**31) or new_biases.max() > 2**31 - 1:
        return None

    new_input_zp = relay.Constant(nd.array(np.int32(0)))
    new_conv_args = (*prev_op.args[:2], new_input_zp, *prev_op.args[3:])
    new_conv_op = relay.qnn.op.conv2d(*new_conv_args, **prev_op.attrs)
    bias_constant = relay.Constant(nd.array(new_biases.astype("int32")))

    # If biases was wrapped in an expand_dims op, we must re-wrap it
    if isinstance(biases_data_op, relay.expr.Call):
        new_biases_op = relay.expand_dims(bias_constant, **biases_data_op.attrs)
    else:
        new_biases_op = bias_constant

    return relay.add(new_conv_op, new_biases_op)


@qnn_requantize_alter_layout.register(["arm_cpu"])
def alter_requantize_layout(attrs, inputs, _tinfos, _out_type):
    """Changes a floating point requantize op to use int64 multiply + shift for microTVM.

    Usually, this is done by QNN legalization. However, microTVM wants to manually choose the
    integer rounding constants in order to:
        (a) Have int32, not int64 constants
        (b) Use a constant rounding shift to skip a memory load.

    Ideally, we would pick these constants in the requantize (or fused) schedule. Unfortunately that
    is not currently possible, so we pick them with `alter_layout` as a hack. This will only work if
    the requantize schedule "plays along" with this hack.
    """

    # Only microTVM Cortex-M boards with DSP use the relevant schedules
    current_target = target.Target.current(allow_none=False)
    if not (current_target.features.has_dsp and "cortex-m" in current_target.mcpu):
        return None

    _, in_scale, _, out_scale, _ = inputs
    in_scale_numpy = in_scale.data.numpy().astype("float64")
    out_scale_scalar = out_scale.data.numpy().item()

    # Shifting by 33 and rounding means shifting by 32, adding 1, and shifting by 1 again. This is
    # useful, because shifting a multiplication product by 32 can be done for "free" with SMMUL
    scales = ((in_scale_numpy / out_scale_scalar) * 2**33).astype("int32")

    # Requantize ops in Relay do not support int32 scales - if we try to use one, requantize.cc will
    # raise an error. As a hacky work-around, we change the scale dtype to float32, without changing
    # underlying data. This works, as our compute function knows to interpret the scale as an int32.

    # This is only a work-around - a better long-term solution would be adding a new integer
    # requantize op, which takes integer scales, shifts, and rounding behavior.
    fake_float_scales = scales.view("float32")

    scale_constant = relay.Constant(nd.array(fake_float_scales))
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    new_attrs["out_dtype"] = "int16"
    return relay.qnn.op.requantize(inputs[0], scale_constant, *inputs[2:], **new_attrs)
