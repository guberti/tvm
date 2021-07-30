#include "src/standalone_crt/src/runtime/crt/microtvm_rpc_server.h"

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  g_num_bytes_requested += size;
  Serial.write(data, size);
  return size;
}

void setup() {
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM Arduino runtime - running");
}

void loop() {
  noInterrupts();
  int available = Serial.available();
  uint8_t data[available];
  size_t bytes_read = Serial.readBytes(data, available);

  if (bytes_read > 0) {
    size_t bytes_remaining = bytes_read;
    while (bytes_remaining > 0) {
      // Pass the received bytes to the RPC server.
      tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &data, &bytes_remaining);
      if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
        TVMPlatformAbort(err);
      }
      if (g_num_bytes_written != 0 || g_num_bytes_requested != 0) {
        if (g_num_bytes_written != g_num_bytes_requested) {
          TVMPlatformAbort((tvm_crt_error_t)0xbeef5);
        }
        g_num_bytes_written = 0;
        g_num_bytes_requested = 0;
      }
    }
  }
  interrupts();
}
