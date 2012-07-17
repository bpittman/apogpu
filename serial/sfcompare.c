/*
** Copyright (C) 2001-2011 Erik de Castro Lopo <erikd@mega-nerd.com>
**
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**
**     * Redistributions of source code must retain the above copyright
**       notice, this list of conditions and the following disclaimer.
**     * Redistributions in binary form must reproduce the above copyright
**       notice, this list of conditions and the following disclaimer in
**       the documentation and/or other materials provided with the
**       distribution.
**     * Neither the author nor the names of any contributors may be used
**       to endorse or promote products derived from this software without
**       specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
** TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
** PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
** CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
** EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
** PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
** OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
** WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
** OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
** ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include	<stdio.h>
#include 	<stdlib.h>
#include 	<math.h>

/* Include this header file to use functions from libsndfile. */
#include	<sndfile.h>


int
main (int argc, char** argv)
{   /* This is a buffer of double precision floating point values
    ** which will hold our data while we process it.
    */
    float *data1, *data2;

    /* A SNDFILE is very much like a FILE in the Standard C library. The
    ** sf_open function return an SNDFILE* pointer when they sucessfully
	** open the specified file.
    */
    SNDFILE      *infile1, *infile2 ;

    /* A pointer to an SF_INFO stutct is passed to sf_open.
    ** On read, the library fills this struct with information about the file.
    ** On write, the struct must be filled in before calling sf_open.
    */
    SF_INFO		sfinfo1, sfinfo2 ;
    int			readcount,i,min ;
    const char	*filename1;
    const char	*filename2;

    /* Here's where we open the input file. We pass sf_open the file name and
    ** a pointer to an SF_INFO struct.
    ** On successful open, sf_open returns a SNDFILE* pointer which is used
    ** for all subsequent operations on that file.
    ** If an error occurs during sf_open, the function returns a NULL pointer.
	**
	** If you are trying to open a raw headerless file you will need to set the
	** format and channels fields of sfinfo before calling sf_open(). For
	** instance to open a raw 16 bit stereo PCM file you would need the following
	** two lines:
	**
	**		sfinfo.format   = SF_FORMAT_RAW | SF_FORMAT_PCM_16 ;
	**		sfinfo.channels = 2 ;
    */
    if(argc < 3) { printf("insufficient args\n"); return 1; }

    filename1 = argv[1];
    filename2 = argv[2];

    printf("%s %s\n",filename1, filename2);;;

    if (! (infile1 = sf_open (filename1, SFM_READ, &sfinfo1)))
    {   /* Open failed so print an error message. */
        printf ("Not able to open input file %s.\n", filename1) ;
        /* Print the error message from libsndfile. */
        puts (sf_strerror (NULL)) ;
        return  1 ;
    } ;

    if (! (infile2 = sf_open (filename2, SFM_READ, &sfinfo2)))
    {   /* Open failed so print an error message. */
        printf ("Not able to open input file %s.\n", filename1) ;
        /* Print the error message from libsndfile. */
        puts (sf_strerror (NULL)) ;
        return  1 ;
    } ;

    printf("%d %d\n",sfinfo1.frames, sfinfo2.frames);

    data1 = (float*) malloc(sizeof(float)*sfinfo1.frames);
    data2 = (float*) malloc(sizeof(float)*sfinfo2.frames);

    readcount = sf_read_float (infile1, data1, sfinfo1.frames);
    readcount = sf_read_float (infile2, data2, sfinfo2.frames);

    if(sfinfo1.frames < sfinfo2.frames) min = sfinfo1.frames;
    else min = sfinfo2.frames;

    for(i=0;i<min;++i) {
        if(fabs(data1[i]-data2[i]) > 0.0001) {
            printf("%d: %f %f %f\n",i,data1[i],data2[i],fabs(data1[i]-data2[i]));
        }
    }

    if(sfinfo1.frames != sfinfo2.frames) printf("number of frames differs: %d %d\n",sfinfo1.frames, sfinfo2.frames);

    /* Close input and output files. */
    sf_close (infile1) ;
    sf_close (infile2) ;

    return 0 ;
} /* main */

