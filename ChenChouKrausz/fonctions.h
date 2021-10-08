/*****************************************************************



*****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <immintrin.h>
#include <gmp.h>
#include "ccount.h"

#ifndef _FONCTIONS_H
#define _FONCTIONS_H

#define WORD 64


/***************************************************************

	Fonctions d'affichage

*/
  

void afficheVect(uint64_t *A, char *var, int size);


/***************************************************************

	Polynomial Multiplication (in GF2[X])

*/


int karat_mult3(uint64_t * C, const uint64_t * A, const uint64_t * B, int size);

#endif
