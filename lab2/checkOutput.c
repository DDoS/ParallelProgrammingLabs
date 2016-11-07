#include "output.h"
#include "outputProg.h"

int main(char argc, char* argv){
  float cumSum = 0;

  int i=0;
  for (i=0 ; i<2000 ; i++){
    cumSum += (output[i]-outputProg[i])*(output[i]-outputProg[i]);
    printf("%f\n",cumSum);
  }
  float result = cumSum/2000;

  printf("\n\n\n%f\n", result);

  return 0;
}