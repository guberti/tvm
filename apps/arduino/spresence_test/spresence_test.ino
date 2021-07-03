#include "Tvmq.h"
#include "src/inputs.c"

void setup() {
  Tvmq t = Tvmq();
  int8_t output_data[10];
  t.inference(input_data_data, output_data);
}

void loop() {
  digitalWrite(13, HIGH);
  delay(50);
  digitalWrite(13, LOW);
  delay(50);
}
