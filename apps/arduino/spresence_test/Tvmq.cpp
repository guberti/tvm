#include "Tvmq.h"
#include "src/standalone_crt/include/tvm/runtime/crt/logging.h"
#include "src/standalone_crt/include/tvm/runtime/crt/crt.h"
#include "src/standalone_crt/include/tvm/runtime/crt/packed_func.h"
#include "src/standalone_crt/include/tvm/runtime/crt/graph_executor.h"
#include "src/standalone_crt/include/dlpack/dlpack.h"

// Model
#include "src/graph_json.c"
#include "Arduino.h"

Tvmq::Tvmq()
{
  tvm_crt_error_t ret = TVMInitializeRuntime();


  TVMPackedFunc pf;
  TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
  TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args);
  TVMPackedFunc_Call(&pf);

  TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

  // Create device
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  DLDevice dev;
  dev.device_type = (DLDeviceType)device_type;
  dev.device_id = device_id;

  graph_runtime = NULL;
  TVMGraphExecutor_Create(graph_json, mod_syslib, &dev, &graph_runtime);
}


void Tvmq::inference(uint8_t input_data[3072], int8_t *output_data) {
  // Reformat input data into tensor
  static const int64_t input_data_shape[4] = {1, 32, 32, 3};
  DLTensor input_data_tensor = {
    (void*) input_data,
    {kDLCPU, 0},
    4,
    {kDLInt, 8, 0},
    (void*) input_data_shape,
    NULL,    0};

  // Run inputs through the model
  TVMGraphExecutor_SetInput(graph_runtime, "input_1_int8", (DLTensor*) &input_data_tensor);
  TVMGraphExecutor_Run(graph_runtime);

  // Prepare our output tensor
  int64_t output_data_shape[2] = {1, 10};
  DLTensor output_data_tensor = {output_data, {kDLCPU, 0}, 2, {kDLInt, 8, 0}, output_data_shape, NULL, 0};
  TVMGraphExecutor_GetOutput(graph_runtime, 0, &output_data_tensor);
}

int Tvmq::infer_category(uint8_t input_data[3072]) {
  int8_t output_data[10] = {0};
  Tvmq::inference(input_data, output_data);
  int best = -1;
  int maximum = -1000;
  Serial.println("Output tensor:");
  for (int i = 0; i < 10; i++) {
    Serial.println(output_data[i]);
    if (output_data[i] > maximum) {
      maximum = output_data[i];
      best = i;
    }
  }
  return best;
}
