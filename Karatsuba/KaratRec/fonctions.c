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
		for(int j=0;j<WORD;j++) tmp^= ((A[size-1-i]>>j)&1UL)<<(WORD-1-j);
		printf("%16.16lX ",tmp);
	}
	printf("\n");
}


#ifdef SB256
	#include "SB256.c"
#elif KARAT256
	#include "KARAT256.c"
#elif DGK
	#include "DGK.c"
#elif AVX2
	#include "AVX2.c"
#elif KARAT512
	#include "KARAT512.c"
#elif DGK2
	#include "DGK2.c"
#elif SB512
	#include "SB512.c"
#elif KARAT512_SB
	#include "KARAT512_SB.c"
#endif
