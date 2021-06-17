#include "Tvmq.h"
#include <stdint.h>
#include "src/automobile.c"
#include "src/airplane.c"
#include "src/bird.c"
#include "src/cat.c"
#include "src/deer.c"
#include "src/dog.c"
#include "src/frog.c"
#include "src/horse.c"
#include "src/ship.c"
#include "src/truck.c"

static char LABELS[10][10] = {
  "AUTOMOBILE",
  "AIRPLANE",
  "BIRD",
  "CAT",
  "DEER",
  "DOG",
  "FROG",
  "HORSE",
  "SHIP",
  "TRUCK",
};

void setup() {
  Serial.begin(9600);
  Serial.println("Starting program");
  Serial.flush();
  Tvmq t = Tvmq();
  Serial.println("Initialized tensor");
  Serial.flush();
  int category;
  category = t.infer_category(input_ship);
  Serial.print("Identified ship as: ");
  Serial.println(LABELS[category]);
  Serial.flush();
  category = t.infer_category(input_cat);
  Serial.print("Identified cat as: ");
  Serial.println(LABELS[category]);
  Serial.flush();
}

void loop() {
  // put your main code here, to run repeatedly:

}
