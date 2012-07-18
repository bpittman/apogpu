#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sndfile.h>

#include "apogpu.h"

#define MAX_CHANNELS 6

int main (int argc, char **argv) {
   static float *data = NULL;

   SNDFILE *infile, *outfile;

   SF_INFO             sfinfo ;
   int                 readcount,i ;
   sf_count_t          buffer_length, padded_buffer_length;
   const char  *infilename = "input.wav" ;
   const char  *outfilename = "output.wav" ;

   if (! (infile = sf_open (infilename, SFM_READ, &sfinfo))) {
      printf ("Not able to open input file %s.\n", infilename);
      puts(sf_strerror (NULL));
      return 1;
   }

   printf("samplerate: %d\n",sfinfo.samplerate);
   buffer_length = sfinfo.frames;

   if(buffer_length % BLOCK_SIZE != 0) 
      padded_buffer_length = buffer_length + (BLOCK_SIZE - buffer_length % BLOCK_SIZE);
   else
      padded_buffer_length = buffer_length;

   data = (float*)malloc(sizeof(float)*padded_buffer_length);
   if(data == NULL)
   {
      printf("malloc failed!\n");
      return 1;
   }

   if (sfinfo.channels > MAX_CHANNELS) {
      printf ("Not able to process more than %d channels\n", MAX_CHANNELS) ;
      return  1 ;
   }

   // Open the output file. 
   if (! (outfile = sf_open (outfilename, SFM_WRITE, &sfinfo))) {
      printf ("Not able to open output file %s.\n", outfilename);
      puts (sf_strerror (NULL));
      return 1;
   }

   // While there are.frames in the input file, read them, process
   // them and write them to the output file.
   //
   readcount = sf_read_float (infile, data, buffer_length);
   if(readcount != buffer_length) {
      printf("Unable to read to buffer. %d %d\n",readcount, buffer_length);
      return 1;;
   }

   for(i=buffer_length;i<padded_buffer_length;++i) {
      data[i] = 0.f;
   }

   printf("%f\n",data[0]);
   gpusetup(data,sfinfo.channels,padded_buffer_length);
   printf("%f\n",data[0]);

   sf_write_float (outfile, data, readcount);

   // Close input and output files. 
   sf_close (infile) ;
   sf_close (outfile) ;

   free(data);

   return 0;
}

