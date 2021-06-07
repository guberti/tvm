#include "Tvmq.h"
#include "src/standalone_crt/include/tvm/runtime/crt/crt.h"
/*#include <src/standalone_crt/include/tvm/runtime/crt/packed_func.h>
#include <src/standalone_crt/include/tvm/runtime/crt/graph_executor.h>
#include <src/standalone_crt/include/dlpack/dlpack.h>

// Model
#include "src/inputs.c.inc"
#include "src/graph_json.c.inc"
*/
Tvmq::Tvmq()
{
  TVMInitializeRuntime();


  /*TVMPackedFunc pf;
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
  TVMGraphExecutor_Create(graph_json, mod_syslib, &dev, &graph_runtime);*/
}



// Public Methods //////////////////////////////////////////////////////////////
// Functions available in Wiring sketches, this library, and other libraries

void Tvmq::inference(uint8_t input_data[3072], uint8_t output_data[10]) {
  /*// Reformat input data into tensor
  static const int64_t input_data_shape[4] = {1, 32, 32, 3};
  static const DLTensor input_data_tensor = {
    (void*) input_data,
    {kDLCPU, 0},
    4,
    {kDLInt, 8, 0},
    (void*) input_data_shape,
    NULL,    0};

  // Prepare our output tensor
  int64_t output_data_shape[2] = {1, 10};
  DLTensor output_data_tensor = {&output_data, {kDLCPU, 0}, 2, {kDLInt, 8, 0}, output_data_shape, NULL, 0};

  // Run inputs through the model
  TVMGraphExecutor_SetInput(graph_runtime, "data", (DLTensor*) &input_data_tensor);
  TVMGraphExecutor_Run(graph_runtime);
  TVMGraphExecutor_GetOutput(graph_runtime, 0, &output_data_tensor);*/
}
