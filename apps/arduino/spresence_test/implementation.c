#ifndef IMPLEMENTATION
#define IMPLEMENTATION

#include "src/standalone_crt/include/tvm/runtime/crt/logging.h"
#include "src/standalone_crt/include/tvm/runtime/crt/crt.h"
#include "src/standalone_crt/include/tvm/runtime/crt/graph_executor.h"
#include "src/standalone_crt/include/tvm/runtime/crt/packed_func.h"

#include "crt_config.h"
#include "debug_print.h"

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return 0;
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  serial_printf("Running TVMPlatformAbort");
  for (;;)
    ;
}

// Heap for use by TVMPlatformMemoryAllocate.

// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  if (num_bytes == 0) {
    num_bytes = sizeof(int);
  }
  *out_ptr = malloc(num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  free(ptr);
  return kTvmErrorNoError;
}

unsigned long g_utvm_start_time;

#define MILLIS_TIL_EXPIRY 200

int g_utvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 1;
  g_utvm_start_time = micros();
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time;
  *elapsed_time_seconds = ((double) g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

unsigned int random_seed = 0;
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; ++i) {
    buffer[i] = (uint8_t)4; // Chosen by fair die roll
                            // Guaranteed to be random
  }
  return kTvmErrorNoError;
}

#endif
