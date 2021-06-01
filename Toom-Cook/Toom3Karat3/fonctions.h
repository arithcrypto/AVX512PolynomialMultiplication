/*****************************************************************



*****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>



#include <gf2x.h>



#include <immintrin.h>
#include <gmp.h>
#include "ccount.h"

#ifndef _FONCTIONS_H
#define _FONCTIONS_H

#define CEIL_DIVIDE(a, b)  ((a/b) + (a % b == 0 ? 0 : 1)) /*!< Divide a by b and ceil the result*/
#define BITMASK(a, size) ((1UL << (a % size)) - 1) /*!< Create a mask*/

#define WORD 64


#define SIZE_N (9*T_3W - 384)


#define LAST64 (SIZE_N >> 6)
#define t (LAST64)




#define VEC_N_256_SIZE_64                     (CEIL_DIVIDE(PARAM_N_MULT, 256) << 2)



#define SIZE_N_64 (SIZE_N>>6)



// typedef for main...
union int512_t{
	uint64_t i64[8];
	__m128i i128[4];
	__m256i i256[2];
	__m512i i512[1];

};

typedef union int512_t int512;

/***************************************************************

	Fonctions d'affichage

*/
  

void printVect(uint64_t *A, char *var, int size);


/***************************************************************

	Polynomial Multiplication (in GF2[X])

*/


void toom_3_mult(__m256i * C, const __m256i * A, const __m256i * B);


void toom_3_mult_AVX512(__m512i * C, const __m512i * A, const __m512i * B);

#endif
