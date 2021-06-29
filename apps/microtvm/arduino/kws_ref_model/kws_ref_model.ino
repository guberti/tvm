#include "src/model.h"
#include "input_yes.c"

static char LABELS[12][12] = {
  "DOWN",
  "GO",
  "LEFT",
  "NO",
  "OFF",
  "ON",
  "RIGHT",
  "STOP",
  "UP",
  "YES",
  "SILENCE",
  "UNKNOWN"
};

static Model model;

int getMaxIndex(int8_t data[12]) {
  int best = -1, maximum = -257;

  for (int i = 0; i < 12; i++) {
    if (data[i] > maximum) {
      maximum = data[i];
      best = i;
    }
  }

  return best;
}

void setup() {
  Serial.begin(9600);
  model = Model();
  
  int8_t output_data[12];
  model.inference(input_yes, output_data);
  int labelIndex = getMaxIndex(output_data);
  Serial.print("Classified as: ");
  Serial.println(LABELS[labelIndex]);
}

void loop() {}
