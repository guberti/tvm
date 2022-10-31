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
"""Test code for tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot.tensordot_int16_impl. We do
not run the code in this test - we only check that for a few common parameter configurations (like
those found in the regular and depthwise convolutions of MobileNetV1) the function emits the code it
is supposed to.

Note that a *lot* of instruction reordering happens during compilation from C to assembly (by GCC or
Clang). I've verified that this instruction reordering happens correctly for all the functions here.
For more details on why the generated code is the way it is, see `tensordot_int16_impl`."""

import textwrap

from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot import tensordot_int16_impl


def test_write_3x3_depthwise_code():
    code = tensordot_int16_impl(48, (3, 3), 1, (1, 1), (True, False))
    assert code == textwrap.dedent(
        """
    #include <arm_nnsupportfunctions.h>
    __STATIC_FORCEINLINE __WEAK int tensordot_opt_x1_int16_w48_3x3_dsp(
        int *output, int *tensor, int *kernel
    ) {
      int sum_0;

      int tensor__y00_x00__y00_x01 = tensor[0];
      int tensor__y00_x02__unknown = tensor[1];
      int tensor__y01_x00__y01_x01 = tensor[24];
      int tensor__y01_x02__unknown = tensor[25];
      int tensor__y02_x00__y02_x01 = tensor[48];
      int tensor__y02_x02__unknown = tensor[49];

      int kernel__y00_x00__y00_x01 = kernel[0];
      int kernel__y00_x02__y01_x00 = kernel[1];
      int kernel__y01_x01__y01_x02 = kernel[2];
      int kernel__y02_x00__y02_x01 = kernel[3];
      int kernel__y02_x02__unknown = kernel[4];

      asm ("smuad %0, %1, %2" : "=r" (sum_0) : "r" (tensor__y00_x00__y00_x01), "r" (kernel__y00_x00__y00_x01));
      sum_0 = __builtin_arm_smlabb(tensor__y00_x02__unknown, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __builtin_arm_smlatb(kernel__y00_x02__y01_x00, tensor__y01_x00__y01_x01, sum_0);
      sum_0 = __builtin_arm_smlatb(tensor__y01_x00__y01_x01, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __builtin_arm_smlatb(kernel__y01_x01__y01_x02, tensor__y01_x02__unknown, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y02_x00__y02_x01, kernel__y02_x00__y02_x01, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y02_x02__unknown, kernel__y02_x02__unknown, sum_0);

      output[0] = sum_0;
      return 0;
    }
    """
    )


def test_odd_width_3x3_depthwise_strides_code():
    """This is the function that would be generated for a 1x4x48x48 NCHW input tensor with "SAME"
    padding and (2, 2) strides, being written into NHWC layout. The layout change is encoded by
    out_stride = 4. This is a common use case seen in MobileNetV1, among others.

    Note that despite the rows not being word-aligned, the *tensor pointer will always be word
    aligned (satisfying this requirement) since y_stride = 2."""

    code = tensordot_int16_impl(49, (3, 3), 2, (2, 4), (True, False))
    assert code == textwrap.dedent(
        """
    #include <arm_nnsupportfunctions.h>
    __STATIC_FORCEINLINE __WEAK int tensordot_opt_x2_int16_w49_3x3_2_4_dsp(
        int *output, int *tensor, int *kernel
    ) {
      int sum_0, sum_1;

      int tensor__y00_x00__y00_x01 = tensor[0];
      int tensor__y00_x02__y00_x03 = tensor[1];
      int tensor__y00_x04__unknown = tensor[2];
      int tensor__unknown__y01_x00 = tensor[24];
      int tensor__y01_x01__y01_x02 = tensor[25];
      int tensor__y01_x03__y01_x04 = tensor[26];
      int tensor__y02_x00__y02_x01 = tensor[49];
      int tensor__y02_x02__y02_x03 = tensor[50];
      int tensor__y02_x04__unknown = tensor[51];

      int kernel__y00_x00__y00_x01 = kernel[0];
      int kernel__y00_x02__y01_x00 = kernel[1];
      int kernel__y01_x01__y01_x02 = kernel[2];
      int kernel__y02_x00__y02_x01 = kernel[3];
      int kernel__y02_x02__unknown = kernel[4];

      asm ("smuad %0, %1, %2" : "=r" (sum_0) : "r" (tensor__y00_x00__y00_x01), "r" (kernel__y00_x00__y00_x01));
      sum_0 = __builtin_arm_smlabb(tensor__y00_x02__y00_x03, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __builtin_arm_smlatt(tensor__unknown__y01_x00, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y01_x01__y01_x02, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y02_x00__y02_x01, kernel__y02_x00__y02_x01, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y02_x02__y02_x03, kernel__y02_x02__unknown, sum_0);
      asm ("smuad %0, %1, %2" : "=r" (sum_1) : "r" (tensor__y00_x02__y00_x03), "r" (kernel__y00_x00__y00_x01));
      sum_1 = __builtin_arm_smlabb(tensor__y00_x04__unknown, kernel__y00_x02__y01_x00, sum_1);
      sum_1 = __builtin_arm_smlatt(tensor__y01_x01__y01_x02, kernel__y00_x02__y01_x00, sum_1);
      sum_1 = __builtin_arm_smlad(tensor__y01_x03__y01_x04, kernel__y01_x01__y01_x02, sum_1);
      sum_1 = __builtin_arm_smlad(tensor__y02_x02__y02_x03, kernel__y02_x00__y02_x01, sum_1);
      sum_1 = __builtin_arm_smlabb(tensor__y02_x04__unknown, kernel__y02_x02__unknown, sum_1);

      output[0] = sum_0;
      output[4] = sum_1;
      return 0;
    }
    """
    )


def test_1x1x8_conv_no_dsp_code():
    """This is the function that would be generated for a 1x48x48x8 NHWC input tensor under
    standard convolution with a 1x1 kernel. This is a common use case seen in MobileNetV1,
    among others. We are generating this code for a non-DSP processor, like a Cortex-M0."""

    code = tensordot_int16_impl(48 * 8, (1, 8), 1, (8, 1), (False, False))
    assert code == textwrap.dedent(
        """
    #include <arm_nnsupportfunctions.h>
    __STATIC_FORCEINLINE __WEAK int tensordot_opt_x1_int16_w384_1x8(
        int *output, int *tensor, int *kernel
    ) {
      int sum_0;

      int tensor__y00_x00__y00_x01 = tensor[0];
      int tensor__y00_x02__y00_x03 = tensor[1];
      int tensor__y00_x04__y00_x05 = tensor[2];
      int tensor__y00_x06__y00_x07 = tensor[3];

      int kernel__y00_x00__y00_x01 = kernel[0];
      int kernel__y00_x02__y00_x03 = kernel[1];
      int kernel__y00_x04__y00_x05 = kernel[2];
      int kernel__y00_x06__y00_x07 = kernel[3];

      asm ("smulbb %0, %1, %2" : "=r" (sum_0) : "r" (tensor__y00_x00__y00_x01), "r" (kernel__y00_x00__y00_x01));
      sum_0 = __builtin_arm_smlatt(tensor__y00_x00__y00_x01, kernel__y00_x00__y00_x01, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y00_x02__y00_x03, kernel__y00_x02__y00_x03, sum_0);
      sum_0 = __builtin_arm_smlatt(tensor__y00_x02__y00_x03, kernel__y00_x02__y00_x03, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y00_x04__y00_x05, kernel__y00_x04__y00_x05, sum_0);
      sum_0 = __builtin_arm_smlatt(tensor__y00_x04__y00_x05, kernel__y00_x04__y00_x05, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y00_x06__y00_x07, kernel__y00_x06__y00_x07, sum_0);
      sum_0 = __builtin_arm_smlatt(tensor__y00_x06__y00_x07, kernel__y00_x06__y00_x07, sum_0);

      output[0] = sum_0;
      return 0;
    }
    """
    )



def test_1x1x8_convolution_code():
    """This is the function that would be generated for a 1x48x48x8 NHWC input tensor under
    standard convolution with a 1x1 kernel. This is a common use case seen in MobileNetV1,
    among others. In this scenario, a very high amount of memory re-use means that summing
    four channels at once makes us faster."""

    code = tensordot_int16_impl(48 * 8, (1, 8), 4, (8, 1), (True, False))
    assert code == textwrap.dedent(
        """
    #include <arm_nnsupportfunctions.h>
    __STATIC_FORCEINLINE __WEAK int tensordot_opt_x4_int16_w384_1x8_8_1_dsp(
        int *output, int *tensor, int *kernel
    ) {
      int sum_0, sum_1, sum_2, sum_3;

      int tensor__y00_x00__y00_x01 = tensor[0];
      int tensor__y00_x02__y00_x03 = tensor[1];
      int tensor__y00_x04__y00_x05 = tensor[2];
      int tensor__y00_x06__y00_x07 = tensor[3];
      int tensor__y00_x08__y00_x09 = tensor[4];
      int tensor__y00_x0a__y00_x0b = tensor[5];
      int tensor__y00_x0c__y00_x0d = tensor[6];
      int tensor__y00_x0e__y00_x0f = tensor[7];
      int tensor__y00_x10__y00_x11 = tensor[8];
      int tensor__y00_x12__y00_x13 = tensor[9];
      int tensor__y00_x14__y00_x15 = tensor[10];
      int tensor__y00_x16__y00_x17 = tensor[11];
      int tensor__y00_x18__y00_x19 = tensor[12];
      int tensor__y00_x1a__y00_x1b = tensor[13];
      int tensor__y00_x1c__y00_x1d = tensor[14];
      int tensor__y00_x1e__y00_x1f = tensor[15];

      int kernel__y00_x00__y00_x01 = kernel[0];
      int kernel__y00_x02__y00_x03 = kernel[1];
      int kernel__y00_x04__y00_x05 = kernel[2];
      int kernel__y00_x06__y00_x07 = kernel[3];

      asm ("smuad %0, %1, %2" : "=r" (sum_0) : "r" (tensor__y00_x00__y00_x01), "r" (kernel__y00_x00__y00_x01));
      sum_0 = __builtin_arm_smlad(tensor__y00_x02__y00_x03, kernel__y00_x02__y00_x03, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y00_x04__y00_x05, kernel__y00_x04__y00_x05, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y00_x06__y00_x07, kernel__y00_x06__y00_x07, sum_0);
      asm ("smuad %0, %1, %2" : "=r" (sum_1) : "r" (tensor__y00_x08__y00_x09), "r" (kernel__y00_x00__y00_x01));
      sum_1 = __builtin_arm_smlad(tensor__y00_x0a__y00_x0b, kernel__y00_x02__y00_x03, sum_1);
      sum_1 = __builtin_arm_smlad(tensor__y00_x0c__y00_x0d, kernel__y00_x04__y00_x05, sum_1);
      sum_1 = __builtin_arm_smlad(tensor__y00_x0e__y00_x0f, kernel__y00_x06__y00_x07, sum_1);
      asm ("smuad %0, %1, %2" : "=r" (sum_2) : "r" (tensor__y00_x10__y00_x11), "r" (kernel__y00_x00__y00_x01));
      sum_2 = __builtin_arm_smlad(tensor__y00_x12__y00_x13, kernel__y00_x02__y00_x03, sum_2);
      sum_2 = __builtin_arm_smlad(tensor__y00_x14__y00_x15, kernel__y00_x04__y00_x05, sum_2);
      sum_2 = __builtin_arm_smlad(tensor__y00_x16__y00_x17, kernel__y00_x06__y00_x07, sum_2);
      asm ("smuad %0, %1, %2" : "=r" (sum_3) : "r" (tensor__y00_x18__y00_x19), "r" (kernel__y00_x00__y00_x01));
      sum_3 = __builtin_arm_smlad(tensor__y00_x1a__y00_x1b, kernel__y00_x02__y00_x03, sum_3);
      sum_3 = __builtin_arm_smlad(tensor__y00_x1c__y00_x1d, kernel__y00_x04__y00_x05, sum_3);
      sum_3 = __builtin_arm_smlad(tensor__y00_x1e__y00_x1f, kernel__y00_x06__y00_x07, sum_3);

      output[0] = sum_0;
      output[1] = sum_1;
      output[2] = sum_2;
      output[3] = sum_3;
      return 0;
    }
    """
    )