#ifndef Tvmq_h
#define Tvmq_h

#include "src/standalone_crt/include/tvm/runtime/crt/graph_executor.h"


class Tvmq
{
  public:
    Tvmq();
    void inference(uint8_t input_data[3072], int8_t *ext_output_data);
    int infer_category(uint8_t input_data[3072]);

  private:
    TVMGraphExecutor* graph_runtime;
};

#endif

