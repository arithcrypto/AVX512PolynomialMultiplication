/*********************************************

	Recursive Karatsuba 3-way Multiplication
	over GF_2[X]
	Verification, Timing and Instruction count


*********************************************/
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <immintrin.h>
#include <time.h>

#include "ccount.h"


#include "fonctions.h"


#define NSAMPLES 50

#define NTEST 1000

unsigned long long int START, STOP, START1,STOP1;

#define SIZE_MAX_TEST 131072 // maximum size for the tests

#define SIZE_MAX_TEST_512 (SIZE_MAX_TEST>>9)


static inline void init_nA_nB(unsigned long int * nA, unsigned long int * nB)
{

	for(int j=0; j<LAST64;j++){
		nA[j] = (((unsigned long int)(rand()+rand())<<32)^(rand()+rand()));
		nB[j] = (((unsigned long int)(rand()+rand())<<32)^(rand()+rand()));
	}
	


}

/********************************************************************************
*
* MAIN
*
*********************************************************************************/



int main(int argc, char* argv[]){


	int flag=0, counter=0;

	static int512 nA[SIZE_MAX_TEST_512], nB[SIZE_MAX_TEST_512], res[2*SIZE_MAX_TEST_512], resMul[2*SIZE_MAX_TEST_512], tmp[2*SIZE_MAX_TEST_512];
	uint64_t mini = (uint64_t)-1L, mini1 = (uint64_t)-1L, tmp64[SIZE_MAX_TEST<<1];
 
	unsigned long long int timer=0, timer1=0, timer2=0;
	
	
	printf("SIZE_N = %d\n",SIZE_N);
	
	printf("SIZE_N = %d, t = %d, t/(256/WORD) =%d\n",SIZE_N,t,t/(256/WORD));
	printf("%d\n",((SIZE_N/WORD) + (SIZE_N%WORD == 0 ? 0 : 1)));
	srand(time(NULL));

	printf("SIZE_N_64 = %d, \n",SIZE_N_64);
	
	printf("SIZE_MAX_TEST_512 = %d, \n",SIZE_MAX_TEST_512);
	
	
		
#ifdef TEST
	goto chrono;
#endif
	
	init_nA_nB(nA->i64,nB->i64);

	printVect(nA->i64,"nA",(SIZE_N >> 6));
	printVect(nB->i64,"nB",(SIZE_N >> 6));


	/***********************************************/
	printf("\ngf2x_mul :\n----------\n");
	
	
	karat_mult3_5(resMul->i64, nA->i64,nB->i64);

	
	gf2x_mul(res->i64,nA->i64,t,nB->i64,t);
		
	printVect(res->i64,"res (gf2x)",SIZE_N_64<<1);
		
	printf("\nComparison with karat_mult3_5 :\n-----------------------------\n");
	
	printVect(resMul->i64,"resMul",SIZE_N_64<<1);
	
	for(int i=0; i<SIZE_N_64<<1;i++) tmp[i>>3].i64[i&0x7] = res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7];
	
	printf("\n");
	printVect(tmp->i64,"cmp",SIZE_N_64<<1);
	
	printf("\n");
	
	for(int i=0; i<SIZE_N_64<<1;i++)
		if(res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7]) flag++;

	printf("flag = %d ; ",flag);
	
	if(!flag) printf("Victory !!!!!!!!!!!!!!!!!!!\n\n");
	else printf("Too bad !!!!!!!!!!!!!!!!!!!\n\n");
	
	flag=0;
	printf("\n\n");
	

	printf("\nComparison with karat_mult3_5_AVX512 :\n-----------------------------\n");
	karat_mult3_5_AVX512(resMul->i64, nA->i64,nB->i64);

	
	printVect(resMul->i64,"resMul",SIZE_N_64<<1);
	
	for(int i=0; i<SIZE_N_64<<1;i++) tmp[i>>3].i64[i&0x7] = res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7];
	
	printf("\n");
	printVect(tmp->i64,"cmp",SIZE_N_64<<1);
	
	printf("\n");
	
	for(int i=0; i<SIZE_N_64<<1;i++)
		if(res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7]) flag++;

	printf("flag = %d ; ",flag);
	
	if(!flag) printf("Victory !!!!!!!!!!!!!!!!!!!\n\n");
	else printf("Too bad !!!!!!!!!!!!!!!!!!!\n\n");
	
	flag=0;
	printf("\n\n");
	//goto fin;
	
chrono:
	
	

	printf("\t  /****************************/\n");
	printf("\t /   Tests on 1000 datasets   /\n");
	printf("\t/****************************/\n\n");
	
	for(int i=0;i<NTEST;i++)
	{

		init_nA_nB(nA->i64,nB->i64);
		

		gf2x_mul(res->i64,nA->i64,t,nB->i64,t);
		karat_mult3_5(resMul->i64, nA->i64,nB->i64);
		
		for(int i=0; i<SIZE_N_64<<1;i++)
			if(res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7]) flag++;
		flag?counter++,flag=0:counter,flag=0;
	
	}
	if(counter) printf("%d errors !\nToo bad !!!!!!!!!!!!!!!!!!!\n\n",counter),counter=0;
	else printf("gf2x vs karat_mult3_5: Victory !!!!!!!!!!!!!!!!!!!\n\n");
	counter=0;

	


	printf("\t  /****************************/\n");
	printf("\t /   Tests on 1000 datasets   /\n");
	printf("\t/****************************/\n\n");
	
	for(int i=0;i<NTEST;i++)
	{

		init_nA_nB(nA->i64,nB->i64);
		

		gf2x_mul(res->i64,nA->i64,t,nB->i64,t);
		karat_mult3_5_AVX512(resMul->i64, nA->i64,nB->i64);
		
		for(int i=0; i<SIZE_N_64<<1;i++)
			if(res[i>>3].i64[i&0x7]^resMul[i>>3].i64[i&0x7]) flag++;
		flag?counter++,flag=0:counter,flag=0;
	
	}
	if(counter) printf("%d errors !\nToo bad !!!!!!!!!!!!!!!!!!!\n\n",counter),counter=0;
	else printf("gf2x vs karat_mult3_5_AVX512: Victory !!!!!!!!!!!!!!!!!!!\n\n");
	counter=0;


	
	
	printf("\t  /*********************/\n");
	printf("\t / Timings !!!!!!!!!!!!/\n");
	printf("\t/*********************/\n\n");
	
	printf("\t\tSIZE_N = %d\n\t\t Size =%d bits\n\n",SIZE_N,SIZE_N);
	

	printf("t = %d, t/(256/WORD) =%d, \n",t,t/(256/WORD));

	printf("SIZE_N = %d,  \n",SIZE_N);
	
	printf("\ngf2x_mul vs karat_mult3_5\n");
	printf("-------------------------\n");

	
	for(int k=0; k<NSAMPLES;k++){
	
		mini = (uint64_t)-1L, mini1 = (uint64_t)-1L;

		init_nA_nB(nA->i64,nB->i64);
		
		for(int i=0;i<NTEST;i++)
		{
			gf2x_mul(res->i64,nA->i64,t,nB->i64,t);
		}
		
		for(int i=0;i<NTEST;i++)
		{
			
			STAMP(START)
			gf2x_mul(res->i64,nA->i64,t,nB->i64,t);
			STAMP(STOP)
			
			if(mini>STOP-START) mini = STOP-START;
		}

		timer += mini;

	}
	
	for(int k=0; k<NSAMPLES;k++){
	
		mini = (uint64_t)-1L, mini1 = (uint64_t)-1L;

		init_nA_nB(nA->i64,nB->i64);
		
		for(int i=0;i<NTEST;i++)
		{
			karat_mult3_5(resMul->i64, nA->i64,nB->i64);
		}
		
		for(int i=0;i<NTEST;i++)
		{
			STAMP(START1)
			karat_mult3_5(resMul->i64, nA->i64,nB->i64);
			STAMP(STOP1)

			if(mini1>STOP1-START1) mini1 = STOP1-START1;
			
		}
		
		timer1 += mini1;
	}
	
	
	for(int k=0; k<NSAMPLES;k++){
	
		mini = (uint64_t)-1L, mini1 = (uint64_t)-1L;

		init_nA_nB(nA->i64,nB->i64);
		
		for(int i=0;i<NTEST;i++)
		{
			karat_mult3_5_AVX512(resMul->i64, nA->i64,nB->i64);
		}
		
		for(int i=0;i<NTEST;i++)
		{
			STAMP(START1)
			karat_mult3_5_AVX512(resMul->i64, nA->i64,nB->i64);
			STAMP(STOP1)
			if(mini1>STOP1-START1) mini1 = STOP1-START1;
			
		}
		timer2 += mini1;
	}
	printf("timer gf2x_mul           : %llu\n",timer/NSAMPLES);	
	
	printf("timer karat_mult3_5        : %llu\n",timer1/NSAMPLES);

	printf("timer karat_mult3_5_AVX512 : %llu\n",timer2/NSAMPLES);	


fin:
	printf("\n");
	printf("Thanks, bye!\n\n");

}
