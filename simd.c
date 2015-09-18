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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

#ifdef __arm__
#include <arm_neon.h>
#else
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

int int_simd_op(char *opType, unsigned long reps)
{
    int i;
    int ia[4], ib[4], ic[4], id[4];
    char choice = (char)opType[0];
    struct timespec start, end;

#ifdef __arm__
    int32x4_t a, b, c, d;
#else
    __m128i a, b, c, d;
#endif

    srand((unsigned int)time(NULL));

    ia[0] = rand();
    ia[1] = rand();
    ia[2] = rand();
    ia[3] = rand();

    ib[0] = rand();
    ib[1] = rand();
    ib[2] = rand();
    ib[3] = rand();

    ic[0] = rand();
    ic[1] = rand();
    ic[2] = rand();
    ic[3] = rand();

    id[0] = rand();
    id[1] = rand();
    id[2] = rand();
    id[3] = rand();

    /* warm-up loop with nop */
    warmup_loop(reps);

    /* measure loop on its own with nop */
    loop_timer_nop(reps);
  
    /* measure loop on its own */
    loop_timer(reps);

    /* load the vector registers */
#ifdef __arm__
    a = vld1q_s32(&ia[0]);
    b = vld1q_s32(&ib[0]);
    c = vld1q_s32(&ic[0]);
    d = vld1q_s32(&id[0]);
#else
    a = _mm_loadu_si128((__m128i *)&ia[0]);
    b = _mm_loadu_si128((__m128i *)&ib[0]);
    c = _mm_loadu_si128((__m128i *)&ic[0]);
    d = _mm_loadu_si128((__m128i *)&id[0]);
#endif
  
    switch(choice) {
    case '+': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
#ifdef __arm__
	    c = vaddq_s32(a, b);
#else
	    c = _mm_add_epi32(a, b);
#endif
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
#ifdef __arm__
	    c = vaddq_s32(a, d);
#else
	    c = _mm_add_epi32(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Integer Addition - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_s32(a, d);
	    b = vaddq_s32(d, c);
#else
	    c = _mm_add_epi32(a, d);
	    b = _mm_add_epi32(d, c);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Addition - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_s32(a, d);
	    b = vaddq_s32(d, c);
	    a = vaddq_s32(b, d);
	    c = vaddq_s32(d, a);
#else
	    c = _mm_add_epi32(a, d);
	    b = _mm_add_epi32(d, c);
	    a = _mm_add_epi32(b, d);
	    c = _mm_add_epi32(d, a);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Addition - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_s32(a, d);
	    b = vaddq_s32(d, c);
	    a = vaddq_s32(b, d);
	    c = vaddq_s32(d, a);
	    b = vaddq_s32(c, d);
#else
	    c = _mm_add_epi32(a, d);
	    b = _mm_add_epi32(d, c);
	    a = _mm_add_epi32(b, d);
	    c = _mm_add_epi32(d, a);
	    b = _mm_add_epi32(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Integer Addition - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_s32(a, d);
	    b = vaddq_s32(d, c);
	    a = vaddq_s32(b, d);
	    c = vaddq_s32(d, a);
	    b = vaddq_s32(c, d);
	    c = vaddq_s32(d, b);
	    a = vaddq_s32(d, c);
	    b = vaddq_s32(a, d);
#else
	    c = _mm_add_epi32(a, d);
	    b = _mm_add_epi32(d, c);
	    a = _mm_add_epi32(b, d);
	    c = _mm_add_epi32(d, a);
	    b = _mm_add_epi32(c, d);
	    c = _mm_add_epi32(d, b);
	    a = _mm_add_epi32(d, c);
	    b = _mm_add_epi32(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Addition - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_s32(a, d);
	    b = vaddq_s32(d, c);
	    a = vaddq_s32(b, d);
	    c = vaddq_s32(d, a);
	    b = vaddq_s32(c, d);
	    c = vaddq_s32(d, b);
	    a = vaddq_s32(d, c);
	    b = vaddq_s32(a, d);
	    c = vaddq_s32(b, d);
	    b = vaddq_s32(c, d);
#else
	    c = _mm_add_epi32(a, d);
	    b = _mm_add_epi32(d, c);
	    a = _mm_add_epi32(b, d);
	    c = _mm_add_epi32(d, a);
	    b = _mm_add_epi32(c, d);
	    c = _mm_add_epi32(d, b);
	    a = _mm_add_epi32(d, c);
	    b = _mm_add_epi32(a, d);
	    c = _mm_add_epi32(b, d);
	    b = _mm_add_epi32(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Addition - 10 ops");	
	
	break;
    }
    case '-': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
#ifdef __arm__
	    c = vsubq_s32(a, b);
#else
	    c = _mm_sub_epi32(a, b);
#endif
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
#ifdef __arm__
	    c = vsubq_s32(a, d);
#else
	    c = _mm_sub_epi32(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_s32(a, d);
	    b = vsubq_s32(d, c);
#else
	    c = _mm_sub_epi32(a, d);
	    b = _mm_sub_epi32(d, c);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_s32(a, d);
	    b = vsubq_s32(d, c);
	    a = vsubq_s32(b, d);
	    c = vsubq_s32(d, a);
#else
	    c = _mm_sub_epi32(a, d);
	    b = _mm_sub_epi32(d, c);
	    a = _mm_sub_epi32(b, d);
	    c = _mm_sub_epi32(d, a);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_s32(a, d);
	    b = vsubq_s32(d, c);
	    a = vsubq_s32(b, d);
	    c = vsubq_s32(d, a);
	    b = vsubq_s32(c, d);
#else
	    c = _mm_sub_epi32(a, d);
	    b = _mm_sub_epi32(d, c);
	    a = _mm_sub_epi32(b, d);
	    c = _mm_sub_epi32(d, a);
	    b = _mm_sub_epi32(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_s32(a, d);
	    b = vsubq_s32(d, c);
	    a = vsubq_s32(b, d);
	    c = vsubq_s32(d, a);
	    b = vsubq_s32(c, d);
	    c = vsubq_s32(d, b);
	    a = vsubq_s32(d, c);
	    b = vsubq_s32(a, d);
#else
	    c = _mm_sub_epi32(a, d);
	    b = _mm_sub_epi32(d, c);
	    a = _mm_sub_epi32(b, d);
	    c = _mm_sub_epi32(d, a);
	    b = _mm_sub_epi32(c, d);
	    c = _mm_sub_epi32(d, b);
	    a = _mm_sub_epi32(d, c);
	    b = _mm_sub_epi32(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_s32(a, d);
	    b = vsubq_s32(d, c);
	    a = vsubq_s32(b, d);
	    c = vsubq_s32(d, a);
	    b = vsubq_s32(c, d);
	    c = vsubq_s32(d, b);
	    a = vsubq_s32(d, c);
	    b = vsubq_s32(a, d);
	    c = vsubq_s32(b, d);
	    b = vsubq_s32(c, d);
#else
	    c = _mm_sub_epi32(a, d);
	    b = _mm_sub_epi32(d, c);
	    a = _mm_sub_epi32(b, d);
	    c = _mm_sub_epi32(d, a);
	    b = _mm_sub_epi32(c, d);
	    c = _mm_sub_epi32(d, b);
	    a = _mm_sub_epi32(d, c);
	    b = _mm_sub_epi32(a, d);
	    c = _mm_sub_epi32(b, d);
	    b = _mm_sub_epi32(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Subtraction - 10 ops");	
	
	break;
    }
    case '*': {
#ifndef __arm__
	fprintf(stderr, "SIMD integer multiplication is not supported on this platform.\n");
	return 1;
#else
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
	    c = vmulq_s32(a, b);
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
	    c = vmulq_s32(a, d);
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = vmulq_s32(a, d);
	    b = vmulq_s32(d, c);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = vmulq_s32(a, d);
	    b = vmulq_s32(d, c);
	    a = vmulq_s32(b, d);
	    c = vmulq_s32(d, a);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = vmulq_s32(a, d);
	    b = vmulq_s32(d, c);
	    a = vmulq_s32(b, d);
	    c = vmulq_s32(d, a);
	    b = vmulq_s32(c, d);
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = vmulq_s32(a, d);
	    b = vmulq_s32(d, c);
	    a = vmulq_s32(b, d);
	    c = vmulq_s32(d, a);
	    b = vmulq_s32(c, d);
	    c = vmulq_s32(d, b);
	    a = vmulq_s32(d, c);
	    b = vmulq_s32(a, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = vmulq_s32(a, d);
	    b = vmulq_s32(d, c);
	    a = vmulq_s32(b, d);
	    c = vmulq_s32(d, a);
	    b = vmulq_s32(c, d);
	    c = vmulq_s32(d, b);
	    a = vmulq_s32(d, c);
	    b = vmulq_s32(a, d);
	    c = vmulq_s32(b, d);
	    b = vmulq_s32(c, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Integer Multiplication - 10 ops");	
	
	break;
#endif
    }
    default: printf("Only possible operation choices for SIMD are '+', '-' and '*'.\n");
    }

    return 0;
}

int float_simd_op(char *opType, unsigned long reps)
{
    int i;
    float ia[4], ib[4], ic[4], id[4];
    char choice = (char)opType[0];
    struct timespec start, end;

#ifdef __arm__
    float32x4_t a, b, c, d;
#else
    __m128 a, b, c, d;
#endif

    srand((unsigned int)time(NULL));

    ia[0] = rand() / 1.0;
    ia[1] = rand() / 1.0;
    ia[2] = rand() / 1.0;
    ia[3] = rand() / 1.0;

    ib[0] = rand() / 1.0;
    ib[1] = rand() / 1.0;
    ib[2] = rand() / 1.0;
    ib[3] = rand() / 1.0;

    ic[0] = rand() / 1.0;
    ic[1] = rand() / 1.0;
    ic[2] = rand() / 1.0;
    ic[3] = rand() / 1.0;

    id[0] = rand() / 1.0;
    id[1] = rand() / 1.0;
    id[2] = rand() / 1.0;
    id[3] = rand() / 1.0;

    /* warm-up loop with nop */
    warmup_loop(reps);

    /* measure loop on its own with nop */
    loop_timer_nop(reps);
  
    /* measure loop on its own */
    loop_timer(reps);

    /* load the vector registers */
#ifdef __arm__
    a = vld1q_f32(&ia[0]);
    b = vld1q_f32(&ib[0]);
    c = vld1q_f32(&ic[0]);
    d = vld1q_f32(&id[0]);
#else
    a = _mm_loadu_ps(&ia[0]);
    b = _mm_loadu_ps(&ib[0]);
    c = _mm_loadu_ps(&ic[0]);
    d = _mm_loadu_ps(&id[0]);
#endif
  
    switch(choice) {
    case '+': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
#ifdef __arm__
	    c = vaddq_f32(a, b);
#else
	    c = _mm_add_ps(a, b);
#endif
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
#ifdef __arm__
	    c = vaddq_f32(a, d);
#else
	    c = _mm_add_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Float Addition - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_f32(a, d);
	    b = vaddq_f32(d, c);
#else
	    c = _mm_add_ps(a, d);
	    b = _mm_add_ps(d, c);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Addition - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_f32(a, d);
	    b = vaddq_f32(d, c);
	    a = vaddq_f32(b, d);
	    c = vaddq_f32(d, a);
#else
	    c = _mm_add_ps(a, d);
	    b = _mm_add_ps(d, c);
	    a = _mm_add_ps(b, d);
	    c = _mm_add_ps(d, a);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Addition - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_f32(a, d);
	    b = vaddq_f32(d, c);
	    a = vaddq_f32(b, d);
	    c = vaddq_f32(d, a);
	    b = vaddq_f32(c, d);
#else
	    c = _mm_add_ps(a, d);
	    b = _mm_add_ps(d, c);
	    a = _mm_add_ps(b, d);
	    c = _mm_add_ps(d, a);
	    b = _mm_add_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Float Addition - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_f32(a, d);
	    b = vaddq_f32(d, c);
	    a = vaddq_f32(b, d);
	    c = vaddq_f32(d, a);
	    b = vaddq_f32(c, d);
	    c = vaddq_f32(d, b);
	    a = vaddq_f32(d, c);
	    b = vaddq_f32(a, d);
#else
	    c = _mm_add_ps(a, d);
	    b = _mm_add_ps(d, c);
	    a = _mm_add_ps(b, d);
	    c = _mm_add_ps(d, a);
	    b = _mm_add_ps(c, d);
	    c = _mm_add_ps(d, b);
	    a = _mm_add_ps(d, c);
	    b = _mm_add_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Addition - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vaddq_f32(a, d);
	    b = vaddq_f32(d, c);
	    a = vaddq_f32(b, d);
	    c = vaddq_f32(d, a);
	    b = vaddq_f32(c, d);
	    c = vaddq_f32(d, b);
	    a = vaddq_f32(d, c);
	    b = vaddq_f32(a, d);
	    c = vaddq_f32(b, d);
	    b = vaddq_f32(c, d);
#else
	    c = _mm_add_ps(a, d);
	    b = _mm_add_ps(d, c);
	    a = _mm_add_ps(b, d);
	    c = _mm_add_ps(d, a);
	    b = _mm_add_ps(c, d);
	    c = _mm_add_ps(d, b);
	    a = _mm_add_ps(d, c);
	    b = _mm_add_ps(a, d);
	    c = _mm_add_ps(b, d);
	    b = _mm_add_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Addition - 10 ops");	
	
	break;
    }
    case '-': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
#ifdef __arm__
	    c = vsubq_f32(a, b);
#else
	    c = _mm_sub_ps(a, b);
#endif
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
#ifdef __arm__
	    c = vsubq_f32(a, d);
#else
	    c = _mm_sub_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Float Subtraction - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_f32(a, d);
	    b = vsubq_f32(d, c);
#else
	    c = _mm_sub_ps(a, d);
	    b = _mm_sub_ps(d, c);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Subtraction - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_f32(a, d);
	    b = vsubq_f32(d, c);
	    a = vsubq_f32(b, d);
	    c = vsubq_f32(d, a);
#else
	    c = _mm_sub_ps(a, d);
	    b = _mm_sub_ps(d, c);
	    a = _mm_sub_ps(b, d);
	    c = _mm_sub_ps(d, a);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Subtraction - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_f32(a, d);
	    b = vsubq_f32(d, c);
	    a = vsubq_f32(b, d);
	    c = vsubq_f32(d, a);
	    b = vsubq_f32(c, d);
#else
	    c = _mm_sub_ps(a, d);
	    b = _mm_sub_ps(d, c);
	    a = _mm_sub_ps(b, d);
	    c = _mm_sub_ps(d, a);
	    b = _mm_sub_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Float Subtraction - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_f32(a, d);
	    b = vsubq_f32(d, c);
	    a = vsubq_f32(b, d);
	    c = vsubq_f32(d, a);
	    b = vsubq_f32(c, d);
	    c = vsubq_f32(d, b);
	    a = vsubq_f32(d, c);
	    b = vsubq_f32(a, d);
#else
	    c = _mm_sub_ps(a, d);
	    b = _mm_sub_ps(d, c);
	    a = _mm_sub_ps(b, d);
	    c = _mm_sub_ps(d, a);
	    b = _mm_sub_ps(c, d);
	    c = _mm_sub_ps(d, b);
	    a = _mm_sub_ps(d, c);
	    b = _mm_sub_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Subtraction - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vsubq_f32(a, d);
	    b = vsubq_f32(d, c);
	    a = vsubq_f32(b, d);
	    c = vsubq_f32(d, a);
	    b = vsubq_f32(c, d);
	    c = vsubq_f32(d, b);
	    a = vsubq_f32(d, c);
	    b = vsubq_f32(a, d);
	    c = vsubq_f32(b, d);
	    b = vsubq_f32(c, d);
#else
	    c = _mm_sub_ps(a, d);
	    b = _mm_sub_ps(d, c);
	    a = _mm_sub_ps(b, d);
	    c = _mm_sub_ps(d, a);
	    b = _mm_sub_ps(c, d);
	    c = _mm_sub_ps(d, b);
	    a = _mm_sub_ps(d, c);
	    b = _mm_sub_ps(a, d);
	    c = _mm_sub_ps(b, d);
	    b = _mm_sub_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Subtraction - 10 ops");	
	
	break;
    }
    case '*': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
#ifdef __arm__
	    c = vmulq_f32(a, b);
#else
	    c = _mm_mul_ps(a, b);
#endif
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
#ifdef __arm__
	    c = vmulq_f32(a, d);
#else
	    c = _mm_mul_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Float Multiplication - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vmulq_f32(a, d);
	    b = vmulq_f32(d, c);
#else
	    c = _mm_mul_ps(a, d);
	    b = _mm_mul_ps(d, c);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Multiplication - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vmulq_f32(a, d);
	    b = vmulq_f32(d, c);
	    a = vmulq_f32(b, d);
	    c = vmulq_f32(d, a);
#else
	    c = _mm_mul_ps(a, d);
	    b = _mm_mul_ps(d, c);
	    a = _mm_mul_ps(b, d);
	    c = _mm_mul_ps(d, a);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Multiplication - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vmulq_f32(a, d);
	    b = vmulq_f32(d, c);
	    a = vmulq_f32(b, d);
	    c = vmulq_f32(d, a);
	    b = vmulq_f32(c, d);
#else
	    c = _mm_mul_ps(a, d);
	    b = _mm_mul_ps(d, c);
	    a = _mm_mul_ps(b, d);
	    c = _mm_mul_ps(d, a);
	    b = _mm_mul_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Float Multiplication - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vmulq_f32(a, d);
	    b = vmulq_f32(d, c);
	    a = vmulq_f32(b, d);
	    c = vmulq_f32(d, a);
	    b = vmulq_f32(c, d);
	    c = vmulq_f32(d, b);
	    a = vmulq_f32(d, c);
	    b = vmulq_f32(a, d);
#else
	    c = _mm_mul_ps(a, d);
	    b = _mm_mul_ps(d, c);
	    a = _mm_mul_ps(b, d);
	    c = _mm_mul_ps(d, a);
	    b = _mm_mul_ps(c, d);
	    c = _mm_mul_ps(d, b);
	    a = _mm_mul_ps(d, c);
	    b = _mm_mul_ps(a, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Multiplication - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
#ifdef __arm__
	    c = vmulq_f32(a, d);
	    b = vmulq_f32(d, c);
	    a = vmulq_f32(b, d);
	    c = vmulq_f32(d, a);
	    b = vmulq_f32(c, d);
	    c = vmulq_f32(d, b);
	    a = vmulq_f32(d, c);
	    b = vmulq_f32(a, d);
	    c = vmulq_f32(b, d);
	    b = vmulq_f32(c, d);
#else
	    c = _mm_mul_ps(a, d);
	    b = _mm_mul_ps(d, c);
	    a = _mm_mul_ps(b, d);
	    c = _mm_mul_ps(d, a);
	    b = _mm_mul_ps(c, d);
	    c = _mm_mul_ps(d, b);
	    a = _mm_mul_ps(d, c);
	    b = _mm_mul_ps(a, d);
	    c = _mm_mul_ps(b, d);
	    b = _mm_mul_ps(c, d);
#endif
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Float Multiplication - 10 ops");	
	
	break;
    }
    default: printf("Only possible operation choices for SIMD are '+', '-' and '*'.\n");
    }

    return 0;
}

int double_simd_op(char *opType, unsigned long reps)
{
#ifdef __arm__
    fprintf(stderr, "Double precision SIMD is not supported on ARM platform\n");
    return 1;
#else
    int i;
    double ia[2], ib[2], ic[2], id[2];
    char choice = (char)opType[0];
    struct timespec start, end;

    __m128d a, b, c, d;

    srand((unsigned int)time(NULL));

    ia[0] = rand() / 1.0;
    ia[1] = rand() / 1.0;

    ib[0] = rand() / 1.0;
    ib[1] = rand() / 1.0;

    ic[0] = rand() / 1.0;
    ic[1] = rand() / 1.0;

    id[0] = rand() / 1.0;
    id[0] = rand() / 1.0;
    
    /* warm-up loop with nop */
    warmup_loop(reps);

    /* measure loop on its own with nop */
    loop_timer_nop(reps);
  
    /* measure loop on its own */
    loop_timer(reps);

    /* load the vector registers */
    a = _mm_loadu_pd(&ia[0]);
    b = _mm_loadu_pd(&ib[0]);
    c = _mm_loadu_pd(&ic[0]);
    d = _mm_loadu_pd(&id[0]);

    switch(choice) {
    case '+': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
	    c = _mm_add_pd(a, b);
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
	    c = _mm_add_pd(a, d);
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Double Addition - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_add_pd(a, d);
	    b = _mm_add_pd(d, c);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Addition - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_add_pd(a, d);
	    b = _mm_add_pd(d, c);
	    a = _mm_add_pd(b, d);
	    c = _mm_add_pd(d, a);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Addition - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_add_pd(a, d);
	    b = _mm_add_pd(d, c);
	    a = _mm_add_pd(b, d);
	    c = _mm_add_pd(d, a);
	    b = _mm_add_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Double Addition - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_add_pd(a, d);
	    b = _mm_add_pd(d, c);
	    a = _mm_add_pd(b, d);
	    c = _mm_add_pd(d, a);
	    b = _mm_add_pd(c, d);
	    c = _mm_add_pd(d, b);
	    a = _mm_add_pd(d, c);
	    b = _mm_add_pd(a, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Addition - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_add_pd(a, d);
	    b = _mm_add_pd(d, c);
	    a = _mm_add_pd(b, d);
	    c = _mm_add_pd(d, a);
	    b = _mm_add_pd(c, d);
	    c = _mm_add_pd(d, b);
	    a = _mm_add_pd(d, c);
	    b = _mm_add_pd(a, d);
	    c = _mm_add_pd(b, d);
	    b = _mm_add_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Addition - 10 ops");	
	
	break;
    }
    case '-': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
	    c = _mm_sub_pd(a, b);
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
	    c = _mm_sub_pd(a, d);
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Double Subtraction - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_sub_pd(a, d);
	    b = _mm_sub_pd(d, c);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Subtraction - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_sub_pd(a, d);
	    b = _mm_sub_pd(d, c);
	    a = _mm_sub_pd(b, d);
	    c = _mm_sub_pd(d, a);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Subtraction - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_sub_pd(a, d);
	    b = _mm_sub_pd(d, c);
	    a = _mm_sub_pd(b, d);
	    c = _mm_sub_pd(d, a);
	    b = _mm_sub_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Double Subtraction - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_sub_pd(a, d);
	    b = _mm_sub_pd(d, c);
	    a = _mm_sub_pd(b, d);
	    c = _mm_sub_pd(d, a);
	    b = _mm_sub_pd(c, d);
	    c = _mm_sub_pd(d, b);
	    a = _mm_sub_pd(d, c);
	    b = _mm_sub_pd(a, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Subtraction - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_sub_pd(a, d);
	    b = _mm_sub_pd(d, c);
	    a = _mm_sub_pd(b, d);
	    c = _mm_sub_pd(d, a);
	    b = _mm_sub_pd(c, d);
	    c = _mm_sub_pd(d, b);
	    a = _mm_sub_pd(d, c);
	    b = _mm_sub_pd(a, d);
	    c = _mm_sub_pd(b, d);
	    b = _mm_sub_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Subtraction - 10 ops");	
	
	break;
    }
    case '*': {
	/* warm-up loop */
	for (i = 0; i < 100; i++) {
	    c = _mm_mul_pd(a, b);
	}

	/* main loops */
	clock_gettime(CLOCK, &start);
	for (i = 0; i < reps; i++) {
	    c = _mm_mul_pd(a, d);
	}
	clock_gettime(CLOCK, &end);

	elapsed_time_hr(start, end, "SIMD Double Multiplication - 1 op");

	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_mul_pd(a, d);
	    b = _mm_mul_pd(d, c);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Multiplication - 2 ops");
	
	reps = reps / 2;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_mul_pd(a, d);
	    b = _mm_mul_pd(d, c);
	    a = _mm_mul_pd(b, d);
	    c = _mm_mul_pd(d, a);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Multiplication - 4 ops");

	reps = reps * 4 / 5;

	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_mul_pd(a, d);
	    b = _mm_mul_pd(d, c);
	    a = _mm_mul_pd(b, d);
	    c = _mm_mul_pd(d, a);
	    b = _mm_mul_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
      
	elapsed_time_hr(start, end, "SIMD Double Multiplication - 5 ops");

	reps = reps * 5 / 8;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_mul_pd(a, d);
	    b = _mm_mul_pd(d, c);
	    a = _mm_mul_pd(b, d);
	    c = _mm_mul_pd(d, a);
	    b = _mm_mul_pd(c, d);
	    c = _mm_mul_pd(d, b);
	    a = _mm_mul_pd(d, c);
	    b = _mm_mul_pd(a, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Multiplication - 8 ops");
      
	reps = reps * 4 / 5;
	
	clock_gettime(CLOCK, &start);
	for(i=0; i<reps; i++){
	    c = _mm_mul_pd(a, d);
	    b = _mm_mul_pd(d, c);
	    a = _mm_mul_pd(b, d);
	    c = _mm_mul_pd(d, a);
	    b = _mm_mul_pd(c, d);
	    c = _mm_mul_pd(d, b);
	    a = _mm_mul_pd(d, c);
	    b = _mm_mul_pd(a, d);
	    c = _mm_mul_pd(b, d);
	    b = _mm_mul_pd(c, d);
	}
	clock_gettime(CLOCK, &end);
	
	elapsed_time_hr(start, end, "SIMD Double Multiplication - 10 ops");	
	
	break;
    }
    default: printf("Only possible operation choices for SIMD are '+', '-' and '*'.\n");
    }

    return 0;
#endif
}
