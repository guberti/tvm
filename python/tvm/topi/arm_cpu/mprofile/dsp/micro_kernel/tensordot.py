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


def get_c_function_name(tensor_w, kernel_dims, split_size, x_strides, options):
    """Gets the C function name of the tensordot function."""
    return (
        f"tensordot_opt_x{split_size}_int16_w{tensor_w}_"
        + f"{kernel_dims[0]}x{kernel_dims[1]}"
        + (f"_{x_strides[0]}_{x_strides[1]}" if split_size > 1 else "")
        + ("_dsp" if options[0] else "")
        + ("_2xkernel" if options[1] else "")
    )

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


def _init_accumulators(split_size):
    var_names = map(lambda x: f"sum_{x:x}", range(split_size))
    joined_var_names = ", ".join(var_names)
    return f"int {joined_var_names} = 0;"


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


def _get_draft_macs(kernel_dims, tensor_halfwords, kernel_halfwords, offset) -> Iterator[Tuple]:
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
            yield f"smla{tensor_half}{kernel_half}", f"tensor__{tensor_var}", f"kernel__{kernel_var}"


def _apply_simd_optimizations(instruction_tuples) -> Iterator[Tuple]:
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


def _no_first_accumulate(instruction_tuples) -> Iterator[Tuple]:
    ins, op1, op2 = next(instruction_tuples)
    yield NO_ACC_PREFIX_CONVERSIONS[ins], op1, op2
    for instruction_tuple in instruction_tuples:
        yield instruction_tuple



def _expand_instruction_tuples(instruction_tuples, index) -> Iterator[str]:
    """Converts a series of (instruction, var1, var2) tuples into lines of C code. Should be simple,
    but we need to work around a series of cryptic bugs while ensuring the compiler makes certain
    optimizations.

    1. Ideally, we would call __builtin_arm functions instead of including inline assembly, as this
       is easier to read and more future proof. However:
        a. Arm GCC apparently *forgot* to include `__builtin_arm_smlabt`, even though
           `__builtin_arm_smlatt`, `__builtin_arm_smlatb`, `__builtin_arm_smlad` and so on all
           exist. These work as expected on Clang - the issue is GCC only.

        b. Calling `__builtin_arm_smlatt` (and `smlatb` and `smlabb`) works fine on real devices.
           However, calling these builtins causes the Corstone300 simulator to freeze and stall. I
           have no clue on why this is - wouldn't these builtins be compiled to assembly? - yet it
           occurs consistently.


    2. Ideally, the compiler would know that the first multiply instruction should *not* accumulate,
       and would automatically replace it with an otherwise identical but non-accumulating
       instruction. Doing this saves us one cycle, as we don't need to load a zero into sum_i.
       However, the compiler (understandably) does not like overwriting instructions we explicitly
       as for, so we must do this ourselves.

    3. Ideally, since we're going to emit several lines of assembly code, we would do it in a single
       `asm` block. However, we *want* the compiler to reorder the instructions and interleave them
       with memory loads, and it can only do this if we specify the instructions as individual non-
       volatile memory loads.
    """
    for instruction, op1, op2 in instruction_tuples:
        if "smla" in instruction:
            if instruction == "smlabt":
                yield f"sum_{index} = __builtin_arm_smlatb({op2}, {op1}, sum_{index});"
            else:
                yield f"sum_{index} = __builtin_arm_{instruction}({op1}, {op2}, sum_{index});"

        else:
            yield f'asm ("{instruction} %0, %1, %2" : "=r" (sum_{index}) : "r" ({op1}), "r" ({op2}));'

def _write_sums_to_memory(num_sums, out_stride) -> Iterator[str]:
    for i in range(num_sums):
        yield f"output[{i * out_stride}] = sum_{i};"


def tensordot_int16_impl(
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
    function_name = get_c_function_name(tensor_w, kernel_dims, split_size, x_strides, options)
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

        #draft_macs_iter = _no_first_accumulate(draft_macs_iter)
        return _expand_instruction_tuples(draft_macs_iter, index)

    multiply_acc_lines = chain.from_iterable(gen_single_loop_macs(i) for i in range(split_size))
    write_out_lines = _write_sums_to_memory(split_size, out_stride)
    for line in multiply_acc_lines:
        print(line)
    def insert_lines(lines):
        return ("\n" + " " * 10).join(lines)

    # __WEAK allows multiple copies of the function to overwrite themselves, saving flash
    return textwrap.dedent(
        f"""
        int {function_name}(
            int *output, int *tensor, int *kernel
        ) {{
          int sum_0;

          int a = tensor[0];
          int b = tensor[1];

          int c = kernel[0];

          // Replace all calls to kernel[1] with this line to make the bug happen
          int d = kernel[1];

          sum_0 = __builtin_arm_smlad(a, c, sum_0);
          sum_0 = b + d + sum_0;

          output[0] = sum_0;
          return 0;
        }}
        """
    )
