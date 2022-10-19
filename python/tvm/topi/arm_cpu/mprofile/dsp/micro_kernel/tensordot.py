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
"""Computes a "jumpy tensordot" operator, which can be used to tensorize many common operators
including regular conv2d, depthwise conv2d, and grouped conv2d provided the data and kernel layouts
are the optimal ones. When groups=1, the optimal data layout is NHWC and kernel layout is OHWI. When
this is a depthwise convolution, the optimal data layout is NCHW and kernel layout is OIHW."""

from itertools import chain
import textwrap
from typing import Iterator, Tuple

import numpy as np

from tvm import te, tir

from .common import num_simd_lanes_per_word


def _is_pow_2(number):
    """Checks if `number` is a power of `2`."""
    return number & (number - 1) == 0 and number > 0


def _count_factorization_2s(number):
    """Returns the number of times `2` appears in the factorization of `number`."""
    assert isinstance(number, int)
    count = 0
    while number % 2 == 0:
        number // 2
        count += 1
    return count


def _get_func_name(in_dtype, tensor_h, jump, tensor_w):
    """Gets the C function name of the tensordot function."""
    return f"tensordot_{in_dtype}_h{tensor_h}_j{jump}_w{tensor_w}"


def _get_int16_opt_func_name(tensor_w, kernel_dims, split_size, x_strides, options):
    """Gets the C function name of the tensordot function."""
    return (
        f"tensordot_opt_x{split_size}_int16_w{tensor_w}_"
        + f"{kernel_dims[0]}x{kernel_dims[1]}"
        + (f"_{x_strides[0]}_{x_strides[1]}" if split_size > 1 else "")
        + ("_dsp" if options[0] else "")
        + ("_2xkernel" if options[1] else "")
    )


def static_kernel_reshape(kernel, tensor_w, strides, simd_lanes):
    """When we get the kernel, it will be in the correct data layout (it will be altered during the
    legalization step). For the int32 dtype, we're done. But for int8 and int16, we still have an
    issue: non-word-aligned memory loads on Cortex-M take way longer! This manifests in two ways:

      1. Let's do depthwise convolution with a `Mx5` kernel, `in_dtype == "int16"`, `tensor_w = 48`,
         and `strides = (1, 2)`. The input tensor will be formatted along word boundries as:
         ```
             0x0000   0x0001 | 0x0002   0x0003 | 0x0004   0x0005 | ...
             0x0030   0x0031 | 0x0032   0x0033 | 0x0034   0x0035 | ...
         ```
         However, the input tensor will look like:
         ```
             0x0000   0x0001 | 0x0002   0x0003 | 0x0004
             0x0005 | 0x0006   0x0007 | 0x0008   0x0009
         ```
         To be fast, we will want to run `SMLAD(0x0030   0x0031, 0x0005 | 0x0006)` and so on when
         convolving the second rows. But since `0x0005 | 0x0006` spans a word boundry, we can't do
         this fast! Starting two bytes to the right doesn't help either, as then `0x0031 | 0x0032`
         spans a word boundry.

         The problem is even worse than it sounds. Suppose `tensor_w = 49` instead. And with the
         `int8` dtype, three in four rows will not start word aligned!


      2. Next, consider when the horizontal stride (in bytes) is *not* a multiple of the number of
         SIMD lanes. This is guaranteed to shift the word alignment for every horizontal stride (and
         sometimes on vertical strides too), leading to unaligned word divides between the input
         tensor and kernel. Note that in practice, this issue rarely happens for regular Conv2Ds,
         and happens *all the time* for depthwise Conv2Ds.


    All solutions to these issues require more flash memory. There are other approaches, but for the
    sake of generality we choose to duplicate the kernel once for each SIMD lane offset. This change
    can be enabled or disabled via autotuning."""

    assert _is_pow_2(simd_lanes)
    kernel_h, kernel_w = kernel.shape

    # We want kernel_w % simd_lanes to equal tensor_w % simd_lanes
    zero_pad_w = (tensor_w - kernel_w) % simd_lanes
    padded_kernel = np.pad(kernel, ((0, 0), (0, zero_pad_w)))

    shift_w = stride_w % simd_lanes
    shift_h = stride_h * tensor_w % simd_lanes
    copy_shift = min(_count_factorization_2s(shift_w), _count_factorization_2s(shift_h))

    num_kernel_copies = simd_lanes // copy_shift
    len_kernel_copies = (kernel_w + zero_pad_w) * kernel_h + copy_shift
    assert (num_kernel_copies * len_kernel_copies) % simd_lanes == 0

    flat_kernel = np.ravel(padded_kernel)
    flat_output = np.zeros((num_kernel_copies * len_kernel_copies))

    # Copy our flattened kernel at each location
    for i in range(num_kernel_copies):
        flat_output[i * len_kernel_copies : (i + 1) * len_kernel_copies] = flat_kernel

    return flat_output.reshape(num_kernel_copies, len_kernel_copies)


def make_intrin_tensordot(slices, strides, tensordot_params):
    """Helper function for constructing tensordot intrinsic. We can't construct the whole thing here
    (as multiple schedules use tensordot and each must build the intrinstic differently) but we can
    build part here to simplify the code."""

    # in_dtype, tensor_h, jump, tensor_w = tensordot_params
    data, kernel, output = slices
    data_strides, kernel_strides = strides

    data_buf = tir.decl_buffer(
        data.shape, data.dtype, name="data", offset_factor=1, strides=data_strides
    )
    kernel_buf = tir.decl_buffer(
        kernel.shape,
        kernel.dtype,
        name="kernel",
        offset_factor=1,
        strides=kernel_strides,
    )
    output_buf = tir.decl_buffer(
        output.shape, output.dtype, name="output", offset_factor=1, strides=[1]
    )

    def intrin_func(ins, outs):
        builder = tir.ir_builder.create()
        builder.emit(
            tir.call_extern(
                "int32",
                _get_func_name(*tensordot_params),
                outs[0].access_ptr("w"),
                ins[0].access_ptr("r"),
                ins[1].access_ptr("r"),
            )
        )
        return builder.get()

    return te.decl_tensor_intrin(
        output.op,
        intrin_func,
        binds={data: data_buf, kernel: kernel_buf, output: output_buf},
    )


def tensordot_impl(in_dtype: str, tensor_h: int, jump: int, tensor_w: int) -> str:
    simd_lanes = num_simd_lanes_per_word(in_dtype)
    assert tensor_w % simd_lanes == 0
    assert jump % simd_lanes == 0

    if in_dtype == "int8":
        inner_loop = """
              uint32_t tensor_c20 = __SXTB16(tensor_batch);
              uint32_t kernel_c20 = __SXTB16(kernel_batch);
              sum = __SMLAD(tensor_c20, kernel_c20, sum);

              uint32_t tensor_c31 = __SXTB16(__ROR(tensor_batch, 8));
              uint32_t kernel_c31 = __SXTB16(__ROR(kernel_batch, 8));
              sum = __SMLAD(tensor_c31, kernel_c31, sum);"""

    elif in_dtype == "int16":
        inner_loop = """
              sum = __SMLAD(tensor_batch, kernel_batch, sum);"""

    elif in_dtype == "int32":
        inner_loop = """
              // Compiles to a single MAC instruction
              sum += tensor_batch * kernel_batch;"""

    else:
        raise ValueError(f"No tensordot implementation exists for dtype '{in_dtype}'!")

    function_name = _get_func_name(in_dtype, tensor_h, jump, tensor_w)
    return textwrap.dedent(
        (
            f"""
        #include <arm_nnsupportfunctions.h>
        __STATIC_FORCEINLINE __WEAK int {function_name}(
            int *out,
            int *tensor,
            int *kernel) {{

          int sum = 0;


          #pragma GCC unroll {tensor_h}
          for (int i = 0; i < {tensor_h}; i++) {{
            #pragma GCC unroll {tensor_w // simd_lanes}
            for (int j = 0; j < {tensor_w // simd_lanes}; j++) {{
              uint32_t tensor_batch = *tensor++;
              uint32_t kernel_batch = *kernel++;
              {inner_loop.strip()}
            }}
            tensor += {jump // simd_lanes};
          }}
          out[0] = sum;
          return 0;
        }}
        #endif
        """
        )
    )


def _init_accumulators(split_size):
    var_names = map(lambda x: f"sum_{x:x}", range(split_size))
    joined_var_names = ", ".join(var_names)
    return f"int {joined_var_names};"


def _get_tensor_halfwords(tensor_w, kernel_dims, split_size, in_stride) -> Iterator:
    kernel_h, kernel_w = kernel_dims
    split_max = (split_size - 1) * in_stride
    for y in range(kernel_h):
        if y * tensor_w % 2 == 1:
            yield None
        for x in range(kernel_w + split_max):
            yield (y, x)
        if (y * tensor_w + kernel_w + split_max) % 2 == 1:
            yield None


def _get_kernel_halfwords(kernel_dims, _has_multi_kernel) -> Iterator:
    kernel_h, kernel_w = kernel_dims
    for y in range(kernel_h):
        for x in range(kernel_w):
            yield (y, x)
    if kernel_h * kernel_w % 2 == 1:
        yield None


def _get_int16_alias(position) -> str:
    if not position:
        return "unknown"
    y, x = position
    return f"y{y:0>2x}_x{x:0>2x}"


def _load_tensor_vars(halfwords, tensor_w) -> Iterator[str]:
    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        y, x = halfwords[i] or halfwords[i + 1]
        tensor_index = (y * tensor_w + x) // 2
        yield f"int tensor__{var_name} = tensor[{tensor_index}];"


def _load_kernel_vars(halfwords) -> Iterator[str]:
    for i in range(0, len(halfwords), 2):
        var_name = "__".join(map(_get_int16_alias, halfwords[i : i + 2]))
        yield f"int kernel__{var_name} = kernel[{i // 2}];"


def _get_draft_macs(kernel_dims, tensor_halfwords, kernel_halfwords, offset) -> Iterator[str]:
    def get_var(y, x, halfwords):
        i = halfwords.index((y, x))
        if i % 2 == 0:
            return f"{_get_int16_alias((y, x))}__{_get_int16_alias(halfwords[i + 1])}", "b"
        else:
            return f"{_get_int16_alias(halfwords[i - 1])}__{_get_int16_alias((y, x))}", "t"

    kernel_h, kernel_w = kernel_dims
    for y in range(kernel_h):
        for x in range(kernel_w):
            tensor_var, tensor_half = get_var(y, x + offset, tensor_halfwords)
            kernel_var, kernel_half = get_var(y, x, kernel_halfwords)
            yield f"smla{tensor_half}{kernel_half}", tensor_var, kernel_var


def _apply_simd_optimizations(instruction_tuples) -> Iterator[str]:
    curr_tuple = next(instruction_tuples, None)
    while curr_tuple:
        next_tuple = next(instruction_tuples, None)
        if not next_tuple:
            yield curr_tuple
            break

        if curr_tuple[1:] == next_tuple[1:]:
            if set([curr_tuple[0], next_tuple[0]]) == set(["smlatt", "smlabb"]):
                yield "smlad", *curr_tuple[1:]
                next_tuple = next(instruction_tuples, None)
            elif set([curr_tuple[0], next_tuple[0]]) == set(["smlatb", "smlabt"]):
                yield "smladx", *curr_tuple[1:]
                next_tuple = next(instruction_tuples, None)
            else:
                yield curr_tuple

        else:
            yield curr_tuple
        curr_tuple = next_tuple


NO_ACC_PREFIX_CONVERSIONS = {
    "smlad": "smuad",
    "smladx": "smuadx",
    "smlatt": "smultt",
    "smlatb": "smultb",
    "smlabt": "smulbt",
    "smlabb": "smulbb",
}


def _expand_instruction_tuples(tuples, index) -> Iterator[str]:
    prefix = f"sum_{index} = __builtin_arm"
    instruction_name, op1, op2 = next(tuples)
    no_accumulate = NO_ACC_PREFIX_CONVERSIONS[instruction_name]
    yield f"{prefix}_{no_accumulate}(tensor__{op1}, kernel__{op2});"
    for instruction_name, op1, op2 in tuples:
        yield f"{prefix}_{instruction_name}(tensor__{op1}, kernel__{op2}, sum_{index});"


def _write_sums_to_memory(num_sums, out_stride) -> Iterator[str]:
    for i in range(num_sums):
        yield f"output[{i * out_stride}] = sum_{i};"


def optimized_int16_tensordot_impl(
    tensor_w: int,
    kernel_dims: Tuple[int, int],
    split_size: int,
    x_strides: Tuple[int, int],
    options: Tuple[bool, bool],
) -> str:
    """Code for a specialized version of tensordot, which computes `split_size` tensordot operations
    at the same time. Only works with `int16`. The generated function takes as input pointers to the
    output, tensor, and kernel, which must be word-aligned. However, the stride can be half a word.
    """
    function_name = _get_int16_opt_func_name(tensor_w, kernel_dims, split_size, x_strides, options)
    in_stride, out_stride = x_strides
    has_dsp, has_multi_kernel = options

    tensor_halfwords = list(_get_tensor_halfwords(tensor_w, kernel_dims, split_size, in_stride))
    kernel_halfwords = list(_get_kernel_halfwords(kernel_dims, has_multi_kernel))
    load_tensor_lines = _load_tensor_vars(tensor_halfwords, tensor_w)
    load_kernel_lines = _load_kernel_vars(kernel_halfwords)

    def gen_single_loop_macs(index):
        draft_macs_iter = _get_draft_macs(
            kernel_dims, tensor_halfwords, kernel_halfwords, index * in_stride
        )
        if has_dsp:
            draft_macs_iter = _apply_simd_optimizations(draft_macs_iter)
            if has_multi_kernel:
                pass
        return _expand_instruction_tuples(draft_macs_iter, index)

    multiply_acc_lines = chain.from_iterable(gen_single_loop_macs(i) for i in range(split_size))
    write_out_lines = _write_sums_to_memory(split_size, out_stride)

    def insert_lines(lines):
        return ("\n" + " " * 10).join(lines)

    # __WEAK allows multiple copies of the function to overwrite themselves, saving flash
    return textwrap.dedent(
        f"""
        #include <arm_nnsupportfunctions.h>
        __STATIC_FORCEINLINE __WEAK int {function_name}(
            int *out, int *tensor, int *kernel
        ) {{
          {_init_accumulators(split_size)}

          {insert_lines(load_tensor_lines)}

          {insert_lines(load_kernel_lines)}

          {insert_lines(multiply_acc_lines)}

          {insert_lines(write_out_lines)}
          return 0;
        }}
        """
    )
