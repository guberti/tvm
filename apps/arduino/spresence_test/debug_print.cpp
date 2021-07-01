#include "Arduino.h"

extern "C" void led_enable() {
  digitalWrite(13, HIGH);
}

extern "C" void led_disable() {
  digitalWrite(13, LOW);
}
