// ensure this library description is only included once
#ifndef Tvmq_h
#define Tvmq_h

#include "src/standalone_crt/include/tvm/runtime/crt/graph_executor.h"


// library interface description
class Tvmq
{
  // user-accessible "public" interface
  public:
    Tvmq();
    void inference(uint8_t input_data[3072], uint8_t output_data[10]);

  // library-accessible "private" interface
  private:
    TVMGraphExecutor* graph_runtime;
};

#endif

