#include "../standalone_crt/include/tvm/runtime/crt/module.h"
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_divide_add_round_cast_clip_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_avg_pool2d_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_softmax(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__2(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape_cast_subtract(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429_(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_cast_subtract(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t _lookup_linked_param(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)fused_divide_add_round_cast_clip_cast,
    (TVMBackendPackedCFunc)fused_nn_avg_pool2d_cast,
    (TVMBackendPackedCFunc)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__1,
    (TVMBackendPackedCFunc)fused_nn_softmax,
    (TVMBackendPackedCFunc)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__2,
    (TVMBackendPackedCFunc)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast,
    (TVMBackendPackedCFunc)fused_reshape_cast_subtract,
    (TVMBackendPackedCFunc)fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_,
    (TVMBackendPackedCFunc)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429_,
    (TVMBackendPackedCFunc)fused_cast_subtract,
    (TVMBackendPackedCFunc)_lookup_linked_param,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\013fused_divide_add_round_cast_clip_cast\000fused_nn_avg_pool2d_cast\000fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__1\000fused_nn_softmax\000fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429__2\000fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast\000fused_reshape_cast_subtract\000fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_\000fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_9959535092109263429_\000fused_cast_subtract\000_lookup_linked_param\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;