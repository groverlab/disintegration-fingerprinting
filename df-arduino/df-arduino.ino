#include <Servo.h>
Servo servo1;
int incomingByte = 0;

void setup() {
  Serial.begin(115200);
  servo1.attach(9);
  servo1.write(30);  // was 40
}

void loop() {
  Serial.println(analogRead(A1));
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    if (incomingByte == '1') {
      servo1.write(60);  // was 40
    }
    if (incomingByte == '2') {
      servo1.write(90);  // was 70
    }
  }
}
