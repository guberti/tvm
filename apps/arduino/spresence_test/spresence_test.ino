#include "Tvmq.h"

void setup() {
  Tvmq t = Tvmq();
  uint8_t input_data[1960] = {0};
  int8_t output_data[4];
  t.inference(input_data, output_data);
  Serial.begin(9600);
  Serial.println("Finished inference");
}

void loop() {
  digitalWrite(13, HIGH);
  delay(50);
  digitalWrite(13, LOW);
  delay(50);
}
