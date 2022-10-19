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
"""Test code for tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot.optimized_tensordot_impl"""

import textwrap

from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel.tensordot import optimized_int16_tensordot_impl


def test_write_3x3_depthwise_code():
    code = optimized_int16_tensordot_impl(48, (3, 3), 1, (1, 1), (True, False))
    assert code == textwrap.dedent(
        """
    #include <arm_nnsupportfunctions.h>
    __STATIC_FORCEINLINE __WEAK int tensordot_opt_x1_int16_w48_3x3_dsp(
        int *out, int *tensor, int *kernel
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

      sum_0 = __builtin_arm_smuad(tensor__y00_x00__y00_x01, kernel__y00_x00__y00_x01);
      sum_0 = __builtin_arm_smlabb(tensor__y00_x02__unknown, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __builtin_arm_smlabt(tensor__y01_x00__y01_x01, kernel__y00_x02__y01_x00, sum_0);
      sum_0 = __builtin_arm_smlatb(tensor__y01_x00__y01_x01, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __builtin_arm_smlabt(tensor__y01_x02__unknown, kernel__y01_x01__y01_x02, sum_0);
      sum_0 = __builtin_arm_smlad(tensor__y02_x00__y02_x01, kernel__y02_x00__y02_x01, sum_0);
      sum_0 = __builtin_arm_smlabb(tensor__y02_x02__unknown, kernel__y02_x02__unknown, sum_0);

      output[0] = sum_0;
      return 0;
    }
    """
    )
