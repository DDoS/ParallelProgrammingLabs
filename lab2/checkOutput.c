#include <stdio.h>

#include "output.h"
#include "outputProg.h"

int main(int argc, char *argv[]) {
  float cumSum = 0;
  for (int i = 0; i < 2000; i++) {
    cumSum += (output[i] - outputProg[i]) * (output[i] - outputProg[i]);
  }
  float result = cumSum / 2000;
  printf("Mean squared error: %f\n", result);
  return 0;
}
