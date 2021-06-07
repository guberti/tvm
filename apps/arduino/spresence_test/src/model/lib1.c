// tvm target: c -keys=cpu -link-params=0 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1
#define TVM_EXPORTS
#include "../standalone_crt/include/tvm/runtime/c_runtime_api.h"
#include "../standalone_crt/include/tvm/runtime/c_backend_api.h"
#include <math.h>
void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_cast_subtract_1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* T_subtract = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 32; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 32; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        ((int16_t*)T_subtract)[((((ax0_ax1_fused * 512) + (ax2 * 16)) + ax3_inner))] = (((int16_t)((int8_t*)placeholder)[((((ax0_ax1_fused * 512) + (ax2 * 16)) + ax3_inner))]) - ((int16_t*)placeholder1)[(0)]);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_cast_subtract(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* T_subtract = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 32; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 32; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 3; ++ax3_inner) {
        ((int16_t*)T_subtract)[((((ax0_ax1_fused * 96) + (ax2 * 3)) + ax3_inner))] = (((int16_t)((int8_t*)placeholder)[((((ax0_ax1_fused * 96) + (ax2 * 3)) + ax3_inner))]) - ((int16_t*)placeholder1)[(0)]);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_rig_6094505598157483839_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* arg8 = (((TVMValue*)args)[8].v_handle);
  int32_t arg8_code = ((int32_t*)arg_type_ids)[(8)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* placeholder7 = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  void* T_subtract = (((DLTensor*)arg8)[0].data);
  void* arg8_shape = (((DLTensor*)arg8)[0].shape);
  void* arg8_strides = (((DLTensor*)arg8)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  if (!(arg8_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)36992, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 34; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 544) + (i2 * 16)) + i3))] = (((((1 <= i0_i1_fused) && (i0_i1_fused < 33)) && (1 <= i2)) && (i2 < 33)) ? ((int16_t*)placeholder1)[(((((i0_i1_fused * 512) + (i2 * 16)) + i3) - 528))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 16; ++rc) {
            ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 544) + (ry * 544)) + (rx * 16)) + ((ax0_ax1_fused_ax2_fused & 31) * 16)) + rc))]) * ((int32_t)((int16_t*)placeholder2)[(((((ry * 768) + (rx * 256)) + (rc * 16)) + ax3))])));
          }
        }
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder3)[(ax3)])) * ((int64_t*)placeholder4)[(ax3)]) + ((int64_t*)placeholder5)[(ax3)]) >> ((int64_t*)placeholder6)[(ax3)])) + 4;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int32_t _3 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder3)[(ax3)])) * ((int64_t*)placeholder4)[(ax3)]) + ((int64_t*)placeholder5)[(ax3)]) >> ((int64_t*)placeholder6)[(ax3)])) + 4;
      int32_t _4 = (_3) < (127) ? (_3) : (127);
      int32_t _5 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t*)placeholder)[(((ax0_ax1_fused_ax2_fused * 16) + ax3))]) + 128)) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t*)placeholder)[(((ax0_ax1_fused_ax2_fused * 16) + ax3))]) + 128))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 4)) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((_4) > (-128) ? (_4) : (-128)))) - 4))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
      int32_t _6 = (_5) < (127) ? (_5) : (127);
      int8_t _7 = (int8_t)((_6) > (-128) ? (_6) : (-128));
      int8_t _8 = (int8_t)127;
      int8_t _9 = (_7) < (_8) ? (_7) : (_8);
      int8_t _10 = (int8_t)-128;
      ((int16_t*)T_subtract)[(((ax0_ax1_fused_ax2_fused * 16) + ax3))] = (((int16_t)((_9) > (_10) ? (_9) : (_10))) - ((int16_t*)placeholder7)[(0)]);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* T_subtract = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg7_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)36992, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 34; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 544) + (i2 * 16)) + i3))] = (((((1 <= i0_i1_fused) && (i0_i1_fused < 33)) && (1 <= i2)) && (i2 < 33)) ? ((int16_t*)placeholder)[(((((i0_i1_fused * 512) + (i2 * 16)) + i3) - 528))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 16; ++rc) {
            ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 544) + (ry * 544)) + (rx * 16)) + ((ax0_ax1_fused_ax2_fused & 31) * 16)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 768) + (rx * 256)) + (rc * 16)) + ax3))])));
          }
        }
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) - 128;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int16_t*)T_subtract)[(((ax0_ax1_fused_ax2_fused * 16) + ax3))] = (((int16_t)((_5) > (_6) ? (_5) : (_6))) - ((int16_t*)placeholder6)[(0)]);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_divide_add_round_cast_clip_cast(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* T_cast = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax1_inner = 0; ax1_inner < 10; ++ax1_inner) {
    int32_t _1 = (int32_t)roundf(((((float*)placeholder)[(ax1_inner)] * 2.560000e+02f) + -1.280000e+02f));
    int32_t _2 = (_1) < (127) ? (_1) : (127);
    ((int8_t*)T_cast)[(ax1_inner)] = ((int8_t)((_2) > (-128) ? (_2) : (-128)));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape_cast_subtract(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* T_subtract = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      ((int16_t*)T_subtract)[(((ax1_outer * 16) + ax1_inner))] = (((int16_t)((int8_t*)placeholder)[(((ax1_outer * 16) + ax1_inner))]) - ((int16_t*)placeholder1)[(0)]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_698566638812936261_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* T_add = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)20736, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 18; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 18; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 576) + (i2 * 32)) + i3))] = (((((1 <= i0_i1_fused) && (i0_i1_fused < 17)) && (1 <= i2)) && (i2 < 17)) ? ((int16_t*)placeholder)[(((((i0_i1_fused * 512) + (i2 * 32)) + i3) - 544))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 256; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 32; ++rc) {
            ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 4) * 576) + (ry * 576)) + (rx * 32)) + ((ax0_ax1_fused_ax2_fused & 15) * 32)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 3072) + (rx * 1024)) + (rc * 32)) + ax3))])));
          }
        }
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) + 4;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int32_t _3 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) + 4;
      int32_t _4 = (_3) < (127) ? (_3) : (127);
      ((int32_t*)T_add)[(((ax0_ax1_fused_ax2_fused * 32) + ax3))] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 4)) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((_4) > (-128) ? (_4) : (-128)))) - 4))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_9686294413541518587_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* arg8 = (((TVMValue*)args)[8].v_handle);
  int32_t arg8_code = ((int32_t*)arg_type_ids)[(8)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* placeholder7 = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  void* T_subtract = (((DLTensor*)arg8)[0].data);
  void* arg8_shape = (((DLTensor*)arg8)[0].shape);
  void* arg8_strides = (((DLTensor*)arg8)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  if (!(arg8_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)30752, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 31; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 31; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 496) + (i2 * 16)) + i3))] = ((int16_t*)placeholder)[((((i0_i1_fused * 512) + (i2 * 16)) + i3))];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 256; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t rc = 0; rc < 16; ++rc) {
        ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((ax0_ax1_fused_ax2_fused >> 4) * 992) + ((ax0_ax1_fused_ax2_fused & 15) * 32)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((rc * 32) + ax3))])));
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) - 17;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int32_t _3 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) - 17;
      int32_t _4 = (_3) < (127) ? (_3) : (127);
      int32_t _5 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) + 17)) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((_4) > (-128) ? (_4) : (-128)))) + 17))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t*)placeholder6)[(((ax0_ax1_fused_ax2_fused * 32) + ax3))];
      int32_t _6 = (_5) < (127) ? (_5) : (127);
      int8_t _7 = (int8_t)((_6) > (-128) ? (_6) : (-128));
      int8_t _8 = (int8_t)127;
      int8_t _9 = (_7) < (_8) ? (_7) : (_8);
      int8_t _10 = (int8_t)-128;
      ((int16_t*)T_subtract)[(((ax0_ax1_fused_ax2_fused * 32) + ax3))] = (((int16_t)((_9) > (_10) ? (_9) : (_10))) - ((int16_t*)placeholder7)[(0)]);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* T_subtract = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg7_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)34848, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 33; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 33; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 528) + (i2 * 16)) + i3))] = (((i0_i1_fused < 32) && (i2 < 32)) ? ((int16_t*)placeholder)[((((i0_i1_fused * 512) + (i2 * 16)) + i3))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 256; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 16; ++rc) {
            ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 4) * 1056) + (ry * 528)) + ((ax0_ax1_fused_ax2_fused & 15) * 32)) + (rx * 16)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 1536) + (rx * 512)) + (rc * 32)) + ax3))])));
          }
        }
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(ax3)])) * ((int64_t*)placeholder3)[(ax3)]) + ((int64_t*)placeholder4)[(ax3)]) >> ((int64_t*)placeholder5)[(ax3)])) - 128;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int16_t*)T_subtract)[(((ax0_ax1_fused_ax2_fused * 32) + ax3))] = (((int16_t)((_5) > (_6) ? (_5) : (_6))) - ((int16_t*)placeholder6)[(0)]);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__2(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* T_subtract = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg7_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)18496, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 17; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 544) + (i2 * 32)) + i3))] = (((i0_i1_fused < 16) && (i2 < 16)) ? ((int16_t*)placeholder)[((((i0_i1_fused * 512) + (i2 * 32)) + i3))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 64; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)256, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ff = 0; ff < 64; ++ff) {
      ((int32_t*)Conv2dOutput)[(ff)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 32; ++rc) {
            ((int32_t*)Conv2dOutput)[(ff)] = (((int32_t*)Conv2dOutput)[(ff)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 3) * 1088) + (ry * 544)) + ((ax0_ax1_fused_ax2_fused & 7) * 64)) + (rx * 32)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 6144) + (rx * 2048)) + (rc * 64)) + ff))])));
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(ax3_inner)]) + ((int64_t)((int32_t*)placeholder2)[(ax3_inner)])) * ((int64_t*)placeholder3)[(ax3_inner)]) + ((int64_t*)placeholder4)[(ax3_inner)]) >> ((int64_t*)placeholder5)[(ax3_inner)])) - 128;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int16_t*)T_subtract)[(((ax0_ax1_fused_ax2_fused * 64) + ax3_inner))] = (((int16_t)((_5) > (_6) ? (_5) : (_6))) - ((int16_t*)placeholder6)[(0)]);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* compute = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)6936, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 34; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 3; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 102) + (i2 * 3)) + i3))] = (((((1 <= i0_i1_fused) && (i0_i1_fused < 33)) && (1 <= i2)) && (i2 < 33)) ? ((int16_t*)placeholder)[(((((i0_i1_fused * 96) + (i2 * 3)) + i3) - 99))] : (int16_t)0);
      }
    }
  }
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1024; ++i0_i1_fused_i2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t i31 = 0; i31 < 16; ++i31) {
      ((int32_t*)Conv2dOutput)[(0)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 3; ++rc) {
            ((int32_t*)Conv2dOutput)[(0)] = (((int32_t*)Conv2dOutput)[(0)] + (((int32_t)((int16_t*)PaddedInput)[(((((((i0_i1_fused_i2_fused >> 5) * 102) + (ry * 102)) + (rx * 3)) + ((i0_i1_fused_i2_fused & 31) * 3)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 144) + (rx * 48)) + (rc * 16)) + i31))])));
          }
        }
      }
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(0)]) + ((int64_t)((int32_t*)placeholder2)[(i31)])) * ((int64_t*)placeholder3)[(i31)]) + ((int64_t*)placeholder4)[(i31)]) >> ((int64_t*)placeholder5)[(i31)])) - 128;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int8_t*)compute)[(((i0_i1_fused_i2_fused * 16) + i31))] = ((_5) > (_6) ? (_5) : (_6));
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_avg_pool2d_cast(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* T_cast = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* tensor = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)256, 0, 32);
  if (tensor == NULL) {
    return -1;
  }
  for (int32_t ax3_init = 0; ax3_init < 64; ++ax3_init) {
    ((int32_t*)tensor)[(ax3_init)] = 0;
  }
  for (int32_t rv0_rv1_fused = 0; rv0_rv1_fused < 64; ++rv0_rv1_fused) {
    for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
      ((int32_t*)tensor)[(ax3)] = (((int32_t*)tensor)[(ax3)] + ((int32_t*)placeholder)[(((rv0_rv1_fused * 64) + ax3))]);
    }
  }
  for (int32_t ax31 = 0; ax31 < 64; ++ax31) {
    ((int8_t*)T_cast)[(ax31)] = ((int8_t)(((int32_t*)tensor)[(ax31)] / 64));
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_softmax(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* T_softmax_norm = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* T_softmax_maxelem = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 2, 32);
  if (T_softmax_maxelem == NULL) {
    return -1;
  }
  void* T_softmax_exp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)40, 2, 32);
  if (T_softmax_exp == NULL) {
    return -1;
  }
  void* T_softmax_expsum = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4, 2, 32);
  if (T_softmax_expsum == NULL) {
    return -1;
  }
  ((float*)T_softmax_maxelem)[(0)] = -3.402823e+38f;
  for (int32_t k = 0; k < 10; ++k) {
    float _1 = ((float*)T_softmax_maxelem)[(0)];
    float _2 = ((float*)placeholder)[(k)];
    ((float*)T_softmax_maxelem)[(0)] = ((_1) > (_2) ? (_1) : (_2));
  }
  for (int32_t i1 = 0; i1 < 10; ++i1) {
    ((float*)T_softmax_exp)[(i1)] = expf((((float*)placeholder)[(i1)] - ((float*)T_softmax_maxelem)[(0)]));
  }
  ((float*)T_softmax_expsum)[(0)] = 0.000000e+00f;
  for (int32_t k1 = 0; k1 < 10; ++k1) {
    ((float*)T_softmax_expsum)[(0)] = (((float*)T_softmax_expsum)[(0)] + ((float*)T_softmax_exp)[(k1)]);
  }
  for (int32_t i11 = 0; i11 < 10; ++i11) {
    ((float*)T_softmax_norm)[(i11)] = (((float*)T_softmax_exp)[(i11)] / ((float*)T_softmax_expsum)[(0)]);
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_softmax_expsum) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_softmax_exp) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_softmax_maxelem) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* T_multiply = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* compute_global = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)40, 0, 32);
  if (compute_global == NULL) {
    return -1;
  }
  for (int32_t x_c_init = 0; x_c_init < 10; ++x_c_init) {
    ((int32_t*)compute_global)[(x_c_init)] = 0;
  }
  for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
    for (int32_t x_c = 0; x_c < 10; ++x_c) {
      ((int32_t*)compute_global)[(x_c)] = (((int32_t*)compute_global)[(x_c)] + (((int32_t)((int16_t*)placeholder)[(k_outer)]) * ((int32_t)((int16_t*)placeholder1)[(((k_outer * 10) + x_c))])));
    }
  }
  for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 10; ++ax1_inner_inner) {
    int32_t _1 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t*)compute_global)[(ax1_inner_inner)] + ((int32_t*)placeholder2)[(ax1_inner_inner)])) << ((int64_t)0)) : ((int64_t)(((int32_t*)compute_global)[(ax1_inner_inner)] + ((int32_t*)placeholder2)[(ax1_inner_inner)]))) * (int64_t)1552512742) + ((int64_t)1 << ((int64_t)((5 + 31) - 1)))) >> ((int64_t)(5 + 31)))) + 24;
    int32_t _2 = (_1) < (127) ? (_1) : (127);
    ((float*)T_multiply)[(ax1_inner_inner)] = (((float)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 24)) * 1.718535e-01f);
  }
  if (TVMBackendFreeWorkspace(1, dev_id, compute_global) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_698566638812936261__1(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* T_add = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)12800, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 10; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 640) + (i2 * 64)) + i3))] = (((((1 <= i0_i1_fused) && (i0_i1_fused < 9)) && (1 <= i2)) && (i2 < 9)) ? ((int16_t*)placeholder)[(((((i0_i1_fused * 512) + (i2 * 64)) + i3) - 576))] : (int16_t)0);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 64; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)256, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ff = 0; ff < 64; ++ff) {
      ((int32_t*)Conv2dOutput)[(ff)] = 0;
      for (int32_t ry = 0; ry < 3; ++ry) {
        for (int32_t rx = 0; rx < 3; ++rx) {
          for (int32_t rc = 0; rc < 64; ++rc) {
            ((int32_t*)Conv2dOutput)[(ff)] = (((int32_t*)Conv2dOutput)[(ff)] + (((int32_t)((int16_t*)PaddedInput)[(((((((ax0_ax1_fused_ax2_fused >> 3) * 640) + (ry * 640)) + (rx * 64)) + ((ax0_ax1_fused_ax2_fused & 7) * 64)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((((ry * 12288) + (rx * 4096)) + (rc * 64)) + ff))])));
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(ax3_inner)]) + ((int64_t)((int32_t*)placeholder2)[(ax3_inner)])) * ((int64_t*)placeholder3)[(ax3_inner)]) + ((int64_t*)placeholder4)[(ax3_inner)]) >> ((int64_t*)placeholder5)[(ax3_inner)])) - 2;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      ((int32_t*)T_add)[(((ax0_ax1_fused_ax2_fused * 64) + ax3_inner))] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) + 2)) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) + 2))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_628077138851371231_(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = ((int32_t*)arg_type_ids)[(3)];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = ((int32_t*)arg_type_ids)[(4)];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = ((int32_t*)arg_type_ids)[(5)];
  void* arg6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg6_code = ((int32_t*)arg_type_ids)[(6)];
  void* arg7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg7_code = ((int32_t*)arg_type_ids)[(7)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  void* placeholder3 = (((DLTensor*)arg3)[0].data);
  void* arg3_shape = (((DLTensor*)arg3)[0].shape);
  void* arg3_strides = (((DLTensor*)arg3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg4)[0].data);
  void* arg4_shape = (((DLTensor*)arg4)[0].shape);
  void* arg4_strides = (((DLTensor*)arg4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg5)[0].data);
  void* arg5_shape = (((DLTensor*)arg5)[0].shape);
  void* arg5_strides = (((DLTensor*)arg5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg6)[0].data);
  void* arg6_shape = (((DLTensor*)arg6)[0].shape);
  void* arg6_strides = (((DLTensor*)arg6)[0].strides);
  void* T_cast = (((DLTensor*)arg7)[0].data);
  void* arg7_shape = (((DLTensor*)arg7)[0].shape);
  void* arg7_strides = (((DLTensor*)arg7)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  if (!(arg4_strides == NULL)) {
  }
  if (!(arg5_strides == NULL)) {
  }
  if (!(arg6_strides == NULL)) {
  }
  if (!(arg7_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)14400, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 15; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        ((int16_t*)PaddedInput)[((((i0_i1_fused * 480) + (i2 * 32)) + i3))] = ((int16_t*)placeholder)[((((i0_i1_fused * 512) + (i2 * 32)) + i3))];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 64; ++ax0_ax1_fused_ax2_fused) {
    void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)256, 0, 32);
    if (Conv2dOutput == NULL) {
      return -1;
    }
    for (int32_t ff = 0; ff < 64; ++ff) {
      ((int32_t*)Conv2dOutput)[(ff)] = 0;
      for (int32_t rc = 0; rc < 32; ++rc) {
        ((int32_t*)Conv2dOutput)[(ff)] = (((int32_t*)Conv2dOutput)[(ff)] + (((int32_t)((int16_t*)PaddedInput)[(((((ax0_ax1_fused_ax2_fused >> 3) * 960) + ((ax0_ax1_fused_ax2_fused & 7) * 64)) + rc))]) * ((int32_t)((int16_t*)placeholder1)[(((rc * 64) + ff))])));
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t)((((((int64_t)((int32_t*)Conv2dOutput)[(ax3_inner)]) + ((int64_t)((int32_t*)placeholder2)[(ax3_inner)])) * ((int64_t*)placeholder3)[(ax3_inner)]) + ((int64_t*)placeholder4)[(ax3_inner)]) >> ((int64_t*)placeholder5)[(ax3_inner)])) + 38;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int32_t _3 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 38)) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 38))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t*)placeholder6)[(((ax0_ax1_fused_ax2_fused * 64) + ax3_inner))];
      int32_t _4 = (_3) < (127) ? (_3) : (127);
      int8_t _5 = (int8_t)((_4) > (-128) ? (_4) : (-128));
      int8_t _6 = (int8_t)127;
      int8_t _7 = (_5) < (_6) ? (_5) : (_6);
      int8_t _8 = (int8_t)-128;
      ((int32_t*)T_cast)[(((ax0_ax1_fused_ax2_fused * 64) + ax3_inner))] = ((int32_t)((_7) > (_8) ? (_7) : (_8)));
    }
    if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
      return -1;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

