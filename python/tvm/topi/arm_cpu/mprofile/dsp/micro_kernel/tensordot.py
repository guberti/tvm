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

import textwrap

import numpy as np

from tvm import te, tir

from .common import num_simd_lanes_per_word


def _is_pow_2(number):
    """Checks if `number` is a power of `2`."""
    return number & (number-1) == 0 and number > 0


def _count_factorization_2s(number):
    """Returns the number of times `2` appears in the factorization of `number`."""
    assert isinstance(number, int)
    count = 0
    while number % 2 == 0:
        number // 2
        count += 1
    return count



def _get_func_name(in_dtype, tensor_h, jump, tensor_w, suffix):
    """Gets the C function name of the tensordot function."""
    return f"tensordot_{in_dtype}_h{tensor_h}_j{jump}_w{tensor_w}_{suffix}"


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

         The problem is even worse than it sounds. Suppose `tensor_w = 49` instead. We cannot easily
         pad the input tensor without taking up time (due to MetaScheduler's padding optimizations),
         so every other row is *guaranteed* to not be word aligned. And with the `int8` dtype, three
         in four rows will not start word aligned!

         Luckily, we have a clever solution - *pad the kernel so it is misaligned in the same way*.
         In our first example (with `in_dtype == "int16"` and `tensor_w = 48`), we add two bytes of
         zeros to the end of each row in the kernel. If instead `tensor_w = 49`, we don't need
         padding at all with the `int16` dtype, though we must pad with two bytes for `int8`.

         Note that doing this padding *always* makes us faster, even though we do more total memory
         loads in some cases and the unaligned access penalty is only one cycle.


      2. The second problem is more straightfoward - consider when the horizontal stride (in bytes)
         is *not* a multiple of the number of SIMD lanes. This is guaranteed to shift the word
         alignment for every horizontal stride (and sometimes on vertical strides too), leading to
         unaligned word divides between the input tensor and kernel. Note that in practice, this
         issue rarely happens for regular Conv2Ds, and happens *all the time* for depthwise Conv2Ds.

         The solution is to create copies of the kernel in memory for every SIMD lane offset. This
         does take up more flash memory, but kernels are small enough this is probably fine. We must
         also take our padding from issue #1 into account here.

         This change *does not* always make us faster, because switching between kernel copies with
         different offsets takes 1-2 cycles (1 normally, 2 when it prevents pipelining). A depthwise
         convolution with `in_dtype == "int8"` and a `1x1` kernel would incur more overhead via this
         mechanism then we'd save by preventing unaligned access. However, this is an edge case no
         one would ever use in practice (IRL they'd use a broadcast multiply), as are the other
         cases where this change hurts performance. Hence, we will do it whenever the strides aren't
         a mulitple of SIMD lanes.


    We could fix this by padding the input tensor, but doing this without using more time (or
    messing up MetaScheduler's padding optimizations) is hard. Luckily, we can fix both by only
    modifying the kernel.

    This function performs these fixes, and works in full generality (to make supporting Arm MVE
    easier). Note that we don't care about the word width and in_dtype - we only care about the
    quotient of their lengths (word_width / in_dtype = simd_lanes)."""

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
        flat_output[i * len_kernel_copies:(i+1) * len_kernel_copies] = flat_kernel

    return flat_output.reshape(num_kernel_copies, len_kernel_copies)


def make_intrin_tensordot(slices, strides, tensordot_params):
    """Helper function for constructing tensordot intrinsic. We can't construct the whole thing here
    (as multiple schedules use tensordot and each must build the intrinstic differently) but we can
    build part here to simplify the code."""

    # in_dtype, tensor_h, jump, tensor_w, suffix = tensordot_params
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


def simple_tensordot_impl(in_dtype: str, tensor_h: int, jump: int, tensor_w: int) -> str:
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

    function_name = _get_func_name(in_dtype, tensor_h, jump, tensor_w, suffix)
    return textwrap.dedent(
        (
            f"""
        #ifndef {function_name.upper()}
        #define {function_name.upper()}
        __STATIC_FORCEINLINE int {function_name}(
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
    var_names = map(lambda x: f"sum_{x:x}", range(num_accumulators))
    return f"int {var_names.join(", ")};"


def _get_tensor_halfwords(tensor_w, kernel_h, kernel_w, split_max):
    for y in range(kernel_h):
        if y * tensor_w % 2 == 1:
            yield None
        for x in range(kernel_w + split_max):
            yield (y, x)
        if (y * tensor_w + kernel_w + split_max) % 2 == 1:
            yield None


def _get_kernel_halfwords(kernel_h, kernel_w):
    for y in kernel_h:
        for x in kernel_w:
            yield (y, x)
    if len(halfword_vars) % 2:
        yield None


def _get_int16_alias(position):
    if not position:
        return "unknown"
    y, x = position
    return f"y{y:0>2x}_x{x:0>2x}"


def _load_tensor_vars(halfwords, tensor_w):
    for i in range(0, len(halfwords), 2):
        var_name = map(_get_int16_alias, halfwords[i:i+2]).join("__")
        y, x = halfwords[i] or halfwords[i+1]
        tensor_index = (y * tensor_w + x) // 2
        return f"int tensor__{var_name} = tensor[{tensor_index}];"


def _load_kernel_vars(halfwords):
    for i in range(0, len(halfwords), 2):
        var_name = map(_get_int16_alias, halfwords[i:i+2]).join("__")
        return f"int kernel__{var_name} = kernel[{i // 2}];"


def _get_draft_macs(kernel_h, kernel_w, tensor_halfwords, kernel_halfwords, offset):
    def get_var_location(y, x, halfwords):
        index = halfwords.index((y, x))
        if index % 2 == 0:
            return f"{_get_int16_alias(y, x)}__{_get_int16_alias(var_names[i + 1])}", "b"
        else:
            return f"{_get_int16_alias(var_names[i - 1])__{_get_int16_alias(y, x)}}", "t"

    for y in range(kernel_h):
        for x in range(kernel_w):
            tensor_var, tensor_half = get_var(y, x + offset, tensor_halfwords)
            kernel_var, kernel_half = get_var(y, x, kernel_halfwords)
            yield f"smla{tensor_half}{kernel_half}", tensor_var, kernel_var


def _apply_simd_optimizations(instruction_tuples):
    i = 0
    while i < len(instruction_tuples):
        if i == len(instruction_tuples) - 1:
            yield instruction_tuples[i]
            break

        curr_ins, *curr_ops = instruction_tuples[i]
        next_ins, *next_ops = instruction_tuples[i + 1]
        if curr_ops == next_ops:
            if set([curr_ins, next_ins]) == set(["smlatt", "smlabb"]):
                yield "smlad", *curr_ops
                i += 2
            elif set([curr_ins, next_ins]) == set(["smlatb", "smlabt"]):
                yield "smladx", *curr_ops
                i += 2
            else:
                yield curr_ins, *curr_ops
                i += 1


NO_ACC_PREFIX_CONVERSIONS = {
    "smlad": "smuad",
    "smladx": "smuadx",
    "smlatt": "smultt",
    "smlatb": "smultb",
    "smlabt": "smulbt",
    "smlann": "smulbb",
}

def _expand_instruction_tuple_lists(tuple_lists, index):
    for j, instruction_tuple in enumerate(instruction_tuples):
        instruction, op1, op2 = instruction_tuple

        if j == 0:
            no_acc_instruction = NO_ACC_PREFIX_CONVERSIONS[instruction]
            yield f"sum_{index} = __builtin_arm_{no_acc_instruction}({op1}, {op2});"
        else:
            yield f"sum_{index} = __builtin_arm_{instruction}({op1}, {op2}, sum_{index});"


def _write_sums_to_memory(num_sums, out_stride):
    for i in range(num_sums):
        yield f"output[{i * out_stride}] = sum_{i};"


def optimized_tensordot_impl(in_dtype, tensor_w, kernel_dims, split_size, in_stride, out_stride, has_dsp, has_offset_kernel) -> str:
    """Code for a specialized version of tensordot, which computes `split_size` tensordot operations
    at the same time. Only works with `int16`. The generated function takes as input pointers to the
    output, tensor, and kernel, which must be word-aligned. However, the stride can be half a word.
    """
    assert in_dtype == "int16"
    kernel_h, kernel_w = kernel_dims

    tensor_halfwords = _get_tensor_halfwords(tensor_w, kernel_h, kernel_w, split_size * stride - 1)
    kernel_halfwords = _get_kernel_halfwords(kernel_h, kernel_w, has_offset_kernel)
    load_tensor_lines = _load_tensor_vars(tensor_halfwords, tensor_w)
    load_kernel_lines = _load_kernel_vars(kernel_halfwords)

    optimized_macs = []
    for offset in range(0, split_size * stride, stride):
        draft_macs_iter = _get_draft_macs(kernel_h, kernel_w, tensor_halfwords, kernel_halfwords, offset)
        if has_dsp:
            draft_macs_iter = _apply_simd_optimizations(list(draft_macs_iter))
            if has_offset_kernel:
                pass

        optimized_macs.extend(_expand_instruction_tuple_lists(draft_macs_iter, offset // stride))

    write_out_lines = _write_sums_to_memory(split_size, out_stride)


    def insert_lines(lines):
        return lines.join(" " * 10 + "\n")

    return textwrap.dedent(
        f"""
        __STATIC_FORCEINLINE int {function_name}(int *out, int *tensor, int *kernel) {{
          {_init_accumulators(split_size)}

          {insert_lines(load_tensor_lines)}

          {insert_lines(load_kernel_lines)}

          {insert_lines(optimized_macs)}

          {insert_lines(write_out_lines)}
          return 0;
        }}
        """
    )
