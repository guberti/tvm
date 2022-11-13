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
import tvm
from tvm import te, tir
from tvm.script import tir as T
from ..utils import get_const_tuple
from ..nn.utils import get_pad_tuple
from ..nn.pad import pad
from .. import tag, nn
from tvm.tir import TensorIntrin
from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel import tensordot


def pick_tensordot_impl(attrs, data, kernel, is_depthwise):
    _, height, width, in_channels = get_const_tuple(data.shape)
    out_channels, kernel_h, kernel_w, _ = get_const_tuple(kernel.shape)
    stride_h, stride_w = get_const_tuple(attrs.strides)

    if is_depthwise:
        assert attrs.data_layout == "NCHW"
        assert attrs.kernel_layout == "OIHW"
        dimensions = (width, kernel_h, kernel_w)
        offsets = (0, 0, 0) # TODO fix
        in_stride = stride_w
    else:
        assert attrs.data_layout == "NHWC"
        assert attrs.kernel_layout == "OHWI"
        dimensions = (width * in_channels, kernel_h, kernel_w * in_channels)
        offsets = (0, 0, 0) # TODO fix
        in_stride = in_channels * stride_w

    assert attrs.out_layout is not None
    if attrs.out_layout == "NHWC":
        out_stride = out_channels
    elif attrs.out_layout == "NCHW":
        out_stride = 1
    else:
        raise ValueError(f"Unsupported output layout {attrs.out_layout}!")

    return (dimensions, offsets, (in_stride, out_stride))


def _get_tscript_const_tuple(values):
    return tuple(tvm.tir.const(n) for n in get_const_tuple(values))


def qnn_conv2d(attrs, inputs, out_type):
    """Compute for qnn.conv2d with NHWC layout. Note that this is a DIFFERENT layout from the
    Hexagon variant, because they have special instructions Cortex-M doesn't have. We also expect
    the kernel to have OHWI layout.
    """
    assert len(inputs) == 11
    data, kernel, _izp, _kzp, _iscale, _kscale, bias, rq_scale = inputs[0:8]

    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    output_layout = attrs.out_layout
    assert output_layout == "NHWC"

    num_sums = 2 # TODO fix
    func_calls_per_row = data.shape[2] // num_sums

    func_hyperparams = pick_tensordot_impl(attrs, data, kernel, False)
    func_name = tensordot.get_c_function_name(num_sums, *func_hyperparams)
    func_code = tensordot.tensordot_int16_impl(num_sums, *func_hyperparams)

    assert rq_scale.dtype == "int32"

    @T.prim_func
    def biased_quantized_conv2d(
        data_handle: T.handle,
        kernel_handle: T.handle,
        bias_handle: T.handle,
        requantize_handle: T.handle,
        output_handle: T.handle,
    ) -> None:

        _batch_size, height, width, in_channels = _get_tscript_const_tuple(data.shape)
        out_channels, kernel_h, kernel_w, _ = _get_tscript_const_tuple(kernel.shape)

        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        DATA = T.match_buffer(data_handle, data.shape, dtype="int16")
        KERNEL = T.match_buffer(kernel_handle, kernel.shape, dtype="int16")
        BIAS = T.match_buffer(bias_handle, bias.shape, dtype="int32")
        REQUANTIZE_SCALE = T.match_buffer(requantize_handle, rq_scale.shape, dtype="int32")
        OUTPUT = T.match_buffer(output_handle, out_type.shape, dtype=out_type.dtype)

        # This hack prevents TVM from seeing these variables as "unused". I should be using T.reads
        # and T.writes, but they don't work. I think it's an issue with BufferTouchedDomain.

        OUTPUT[0, 0, 0, 0] = 0
        x = DATA[0, 0, 0, 0]
        y = KERNEL[0, 0, 0, 0]
        for oh, ow, oc in T.grid(height, T.floordiv(width, 2), out_channels):
            with T.block("conv2d"):
                T.block_attr({"pragma_import_c": func_code})
                voh, vow, voc = T.axis.remap("SSS", [oh, ow, oc])
                T.evaluate(
                    T.call_extern(
                        func_name,
                        T.tvm_access_ptr(
                            T.type_annotation(dtype="int16"),
                            OUTPUT.data,
                            voh * width * out_channels + vow * out_channels * 2 + voc,
                            0,
                            1,
                            dtype="handle",
                        ),
                        T.tvm_access_ptr(
                            T.type_annotation(dtype="int16"),
                            DATA.data,
                            voh * width * in_channels + vow * in_channels * 2,
                            0,
                            1,
                            dtype="handle",
                        ),
                        T.tvm_access_ptr(
                            T.type_annotation(dtype="int16"),
                            KERNEL.data,
                            voc * in_channels,
                            0,
                            1,
                            dtype="handle",
                        ),
                        BIAS[0, 0, 0, voc],
                        REQUANTIZE_SCALE[voc],
                        dtype="int32",
                    )
                )

    output = te.extern_primfunc(
        [data, kernel, bias, rq_scale], biased_quantized_conv2d, name="tir", dtype="int16"
    )
    return [output]


def schedule_qnn_conv2d(attrs, outs, target):
    return None
