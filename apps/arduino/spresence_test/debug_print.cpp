#include "Arduino.h"

extern "C" void serial_write(const char* data) {
  Serial.write(data);
  Serial.flush();
}

extern "C" void serial_printf(const char* msg, ...) {
  char buf[256];
  va_list args;
  va_start(args, msg);
  vsnprintf(buf, 256, msg, args);
  Serial.write(buf);
  va_end(args);
  Serial.flush();
}
