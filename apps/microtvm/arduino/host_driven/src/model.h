#ifndef IMPLEMENTATION_H_
#define IMPLEMENTATION_H_

#define WORKSPACE_SIZE $workspace_size_bytes

#ifdef __cplusplus
extern "C" {
#endif


#include "standalone_crt/include/tvm/runtime/crt/logging.h"

void TVMLogf(const char* msg, ...);
void TVMPlatformAbort(tvm_crt_error_t error);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IMPLEMENTATION_H_
