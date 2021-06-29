#include "src/model.h"

static Model model;

void setup() {
  Serial.begin(9600);
  model = Model();
  int8_t input_data[3072] = {0};
  int8_t output_data[10] = {0};
  model.inference(input_data, output_data);
  Serial.println("Performed inference!");
}

void loop() {
  // put your main code here, to run repeatedly:

}
