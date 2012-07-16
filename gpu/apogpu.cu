#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sndfile.h>

#include "apogpu.h"

int main (int argc, char **argv) {
   int dev_id=0;
   // check for GPU device -- output dev info if found
   if (DeviceSelect(dev_id)<0) { 
      fprintf(stderr,"Error: No GPU Device Not Found\n");
      exit(1);
   }

   // output general information about CUDA Device that is selected
   DeviceInfo(dev_id);

   int *data;
   data = (int*)malloc(sizeof(int)*10);

   gpusetup(data,1,10);

   free(data);
   return 0;
}



