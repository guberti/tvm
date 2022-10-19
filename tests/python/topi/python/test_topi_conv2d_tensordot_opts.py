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
"""Test code for tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot.static_kernel_reshape"""

import numpy as np
import pytest

from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot import static_kernel_reshape


def test_requires_pow_2():
    """simd_lanes must be a power of 2"""
    with pytest.raises(Exception):
        static_kernel_reshape(np.array([2]), 48, (1, 1), 5)


class Test3x3DepthwiseKernelTwoLanes:
    no_pad_no_copy = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    yes_pad_no_copy = np.array([[1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]])
    no_pad_yes_copy = np.array([
        [1, 2] + [3, 4] + [5, 6] + [7, 8] + [9],
           [1] + [2, 3] + [4, 5] + [6, 7] + [8, 9]
    ])
    yes_pad_yes_copy = np.array([
            [1, 2] + [3, 0] + [4, 5] + [6, 0] + [7, 8] + [9, 0] + [0],
               [1] + [2, 3] + [0, 4] + [5, 6] + [0, 7] + [8, 9] + [0, 0]
    ])

    @pytest.mark.parametrize("tensor_w, strides, expected", [
        (48, (1, 1), yes_pad_yes_copy),
        (48, (1, 2), yes_pad_no_copy),
        (48, (2, 1), yes_pad_yes_copy),
        (48, (2, 2), yes_pad_no_copy),
        (49, (1, 1), no_pad_yes_copy),
        (49, (1, 2), no_pad_yes_copy),
        (49, (2, 1), no_pad_yes_copy),
        (49, (2, 2), no_pad_no_copy),
    ])
    def test_kernel_reshape(self, tensor_w, strides, expected):
        print("Running a kernel reshape test!")
        print(f"tensor_w: {tensor_w}")
        print(f"strides: {strides}")
        kernel = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        result = static_kernel_reshape(kernel, tensor_w, strides, 2)
        assert np.array_equal(result, expected)


class Test1x1KernelTwoLanes:
    """Since it is better to use int16 operations on the Arm M4/M7 cores, these are the only tests
    we care about for those cores. All other tests only matter for cores with MVE."""

    no_copy = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    yes_copy = np.array([
        [1, 2] + [3, 4] + [5, 6] + [7, 8] + [0],
           [1] + [2, 3] + [4, 5] + [6, 7] + [8, 0],
    ])

    @pytest.mark.parametrize("tensor_w, strides, expected", [
        (48, (1, 1), yes_copy),
        (48, (1, 2), no_copy),
        (48, (2, 1), yes_copy),
        (48, (2, 2), no_copy),
        (49, (1, 1), yes_copy),
        (49, (1, 2), yes_copy),
        (49, (2, 1), yes_copy),
        (49, (2, 2), no_copy),
    ])
    def test_kernel_reshape(self, tensor_w, strides, expected):
        print("Running a kernel reshape test!")
        print(f"tensor_w: {tensor_w}")
        print(f"strides: {strides}")
        kernel = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
        result = static_kernel_reshape(kernel, tensor_w, strides, 2)
        assert np.array_equal(result, expected)
