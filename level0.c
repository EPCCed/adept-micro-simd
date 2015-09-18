/* Copyright (c) 2015 The University of Edinburgh. */

/* 
* This software was developed as part of the                       
* EC FP7 funded project Adept (Project ID: 610490)                 
* www.adept-project.eu                                            
*/

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "level0.h"

/* Level 0 benchmark driver - calls appropriate function */
/* based on command line arguments.                      */
void bench_level0(char *b, unsigned long r, char *o, char *dt){

  /* SIMD operations */
  if(strcmp(b, "simd") == 0){

    if(strcmp(dt, "int") == 0)
      int_simd_op(o,r);

    else if(strcmp(dt, "float") == 0)
      float_simd_op(o,r);

    else if(strcmp(dt, "double") == 0)
      double_simd_op(o,r);

    else fprintf(stderr, "ERROR: check you are using a valid data type...\n");
  }

  /* everything that has not been implemented */
  else
    printf("Not implemented yet.\n");

}
