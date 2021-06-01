/*****************************************************************



*****************************************************************/



#include "fonctions.h"



void printVect(unsigned long int *A, char *var, int size)
{
	int i;
	unsigned long int tmp;
	printf("%s := ",var);
	
	for(i=0;i<size;i++){
		tmp=0;
		for(int j=0;j<WORD;j++) tmp^= ((A[i]>>j)&1UL)<<(WORD-1-j);
		printf("%16.16lX ",tmp);
	}
	printf("\n");
}

#include "ToomCookMult_AVX2.c"
#include "ToomCookMult_AVX512.c"



