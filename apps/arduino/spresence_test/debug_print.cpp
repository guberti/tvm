#include "Arduino.h"

static bool do_debug = false;

extern "C" void enable_debug() {
  do_debug = true;
}

extern "C" void led_enable() {
  if (do_debug) {
    digitalWrite(13, HIGH);
    delay(500);
  }
}

extern "C" void led_disable() {
  if (do_debug) {
    digitalWrite(13, LOW);
    delay(500);
  }
}

extern "C" void rapid_blink() {
  if (do_debug) {
    digitalWrite(13, HIGH);
    delay(100);
    digitalWrite(13, LOW);
    delay(100);
  }
}
