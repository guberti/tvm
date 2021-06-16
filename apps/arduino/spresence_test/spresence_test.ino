#include "Tvmq.h"
#include <stdint.h>
/*#include "src/automobile.c"
#include "src/airplane.c"
#include "src/bird.c"
#include "src/cat.c"
#include "src/deer.c"
#include "src/dog.c"
#include "src/frog.c"
#include "src/horse.c"
#include "src/inputs.c"
//#include "src/truck.c"*/
#include "src/inputs.c"


void setup() {
  Serial.begin(9600);
  Serial.println("Starting program");
  Serial.flush();
  Tvmq t = Tvmq();
  Serial.println("Initialized tensor");
  Serial.flush();
  int8_t output_data[10];
  t.inference(input_data_data, output_data);

  for (int i = 0; i < 10; i++) {
    Serial.println(output_data[i]);
  }
}

void loop() {
  // put your main code here, to run repeatedly:

}
