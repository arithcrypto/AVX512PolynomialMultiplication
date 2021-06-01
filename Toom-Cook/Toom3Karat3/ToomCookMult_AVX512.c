#define T_TM3_3W (SIZE_N / 3)
#define T_TM3 (SIZE_N + 384)
#define tTM3 ((T_TM3) / WORD)
#define T_TM3_3W_256 ((T_TM3_3W + 128) / (4 * WORD))
#define T_TM3_3W_512 ((T_TM3_3W + 128) / (8 * WORD))
#define T_TM3_3W_64 (T_TM3_3W_256 << 2)

#define T_3W_512 (T_3W >> 9)

#define T2_3W_512 (2 * T_3W_512)


#include "gf2xmul_AVX512.c"


/**
 * @brief Compute B(x) = A(x)/(x+1) 
 *
 * This function computes A(x)/(x+1) using a Quercia like algorithm
 * @param[out] out Pointer to the result
 * @param[in] in Pointer to the polynomial A(x)
 * @param[in] size used to define the number of coeeficients of A
 */


static inline void divByXplus1_512(__m512i* out,__m512i* in,int size){
	uint64_t * A = (uint64_t *) in;
	uint64_t * B = (uint64_t *) out;
	
	B[0] = A[0];
	for(int i=1;i<2*(size<<3);i++)
		B[i]= B[i-1]^A[i];
}


/**
 * @brief Compute C(x) = A(x)*B(x) using TOOM3Mult 
 *
 * This function computes A(x)*B(x) using TOOM-COOK3 Multiplication
 * @param[out] Out Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

void toom_3_mult_AVX512(__m512i * Out, const __m512i * A512, const __m512i * B512)
{
	static __m512i U0[T_TM3_3W_512], V0[T_TM3_3W_512], U1[T_TM3_3W_512], V1[T_TM3_3W_512], U2[T_TM3_3W_512], V2[T_TM3_3W_512];
	
	static __m512i W0[2*(T_TM3_3W_512)], W1[2*(T_TM3_3W_512)], W2[2*(T_TM3_3W_512)], W3[2*(T_TM3_3W_512)], W4[2*(T_TM3_3W_512)];
	static __m512i tmp[4*(T_TM3_3W_512)];// 
	
	static __m512i ro512[6*(T_TM3_3W_512)];
 
	const __m512i zero = (__m512i){0ul,0ul,0ul,0ul,0ul,0ul,0ul,0ul};
	
	uint64_t *A = (uint64_t *)A512;
	uint64_t *B = (uint64_t *)B512;
	
	
	int T2 = T_TM3_3W_64<<1;
	for(int i=0;i<T_TM3_3W_512-1;i++)
	{
		
		int i4 = i<<3;
		int i42 = i4-2;
		U0[i]= _mm512_loadu_si512((void const *)(& A[i4]));
		V0[i]= _mm512_loadu_si512((void const *)(& B[i4]));
		U1[i]= _mm512_loadu_si512((void const *)(& A[i42+T_TM3_3W_64]));
		V1[i]= _mm512_loadu_si512((void const *)(& B[i42+T_TM3_3W_64]));
		U2[i]= _mm512_loadu_si512((void const *)(& A[i4+T2-4]));
		V2[i]= _mm512_loadu_si512((void const *)(& B[i4+T2-4]));
	}
	
	for(int i=T_TM3_3W_512-1;i<T_TM3_3W_512;i++)
	{
		int i4 = i<<3;
		int i41 = i4+1;
		int i42 = i4+2;
		int i43 = i4+3;
		int i44 = i4+4;
		int i45 = i4+5;
		
		
		U0[i]= (__m512i){A[i4],A[i41],A[i42],A[i43],A[i44],A[i45],0x0ul,0x0ul};
		V0[i]= (__m512i){B[i4],B[i41],B[i42],B[i43],B[i44],B[i45],0x0ul,0x0ul};
		
		i4 += T_TM3_3W_64-2;
		i41 = i4+1;
		i42 = i4+2;
		i43 = i4+3;
		i44 = i4+4;
		i45 = i4+5;
		
		U1[i]= (__m512i){A[i4],A[i41],A[i42],A[i43],A[i44],A[i45],0x0ul,0x0ul};
		V1[i]= (__m512i){B[i4],B[i41],B[i42],B[i43],B[i44],B[i45],0x0ul,0x0ul};

		i4 += T_TM3_3W_64-2;
		i41 = i4+1;
		i42 = i4+2;
		i43 = i4+3;
		i44 = i4+4;
		i45 = i4+5;

		U2[i]= (__m512i){A[i4],A[i41],A[i42],A[i43],A[i44],A[i45],0x0ul,0x0ul};
		V2[i]= (__m512i){B[i4],B[i41],B[i42],B[i43],B[i44],B[i45],0x0ul,0x0ul};

	}
	
	// EVALUATION PHASE : x= X^64
	// P(X): P0=(0); P1=(1); P2=(x); P3=(1+x); P4=(\infty)
	// Evaluation: 5*2 add, 2*2 shift; 5 mul (n)
	
	
	//W3 = U2 + U1 + U0 ; W2 = V2 + V1 + V0

	for(int i=0;i<T_TM3_3W_512;i++)
	{
		W3[i]=U0[i]^U1[i]^U2[i];
		W2[i]=V0[i]^V1[i]^V2[i];
	}
	
	//W1 = W2 * W3
	
	
	karat_mult3_m512i( W1, W2, W3);
	
	
	
	//W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x (SIZE = T_TM3_3W_512 !)
	
	uint64_t * U1_64 = ((uint64_t *) U1);
	uint64_t * U2_64 = ((uint64_t *) U2);
	
	uint64_t * V1_64 = ((uint64_t *) V1);
	uint64_t * V2_64 = ((uint64_t *) V2);
	
	W0[0] = (__m512i){0ul,U1_64[0],U1_64[1]^U2_64[0],U1_64[2]^U2_64[1],U1_64[3]^U2_64[2],
						U1_64[4]^U2_64[3],U1_64[5]^U2_64[4],U1_64[6]^U2_64[5]};
	W4[0] = (__m512i){0ul,V1_64[0],V1_64[1]^V2_64[0],V1_64[2]^V2_64[1],V1_64[3]^V2_64[2],
						V1_64[4]^V2_64[3],V1_64[5]^V2_64[4],V1_64[6]^V2_64[5]};
	
	U1_64 = ((uint64_t *) U1)-1;
	U2_64 = ((uint64_t *) U2)-2;
	
	V1_64 = ((uint64_t *) V1)-1;
	V2_64 = ((uint64_t *) V2)-2;
	
	for(int i=1;i<T_TM3_3W_512;i++)
	{
		int i4 = i<<3;
		W0[i] = _mm512_loadu_si512((void const *)(& U1_64[i4]));
		W0[i] ^= _mm512_loadu_si512((void const *)(& U2_64[i4]));
		
		W4[i] = _mm512_loadu_si512((void const *)(& V1_64[i4]));
		W4[i] ^= _mm512_loadu_si512((void const *)(& V2_64[i4]));
	}
	
	

	
	//W3 = W3 + W0      ; W2 = W2 + W4
	for(int i=0;i<T_TM3_3W_512;i++)
	{
		W3[i] ^= W0[i];
		W2[i] ^= W4[i];
	}
	

	//W0 = W0 + U0      ; W4 = W4 + V0
	for(int i=0;i<T_TM3_3W_512;i++)
	{
		W0[i] ^= U0[i];
		W4[i] ^= V0[i];
	}


	//W3 = W3 * W2      ; W2 = W0 * W4
	karat_mult3_m512i( tmp, W3, W2);
	for(int i=0;i<2*(T_TM3_3W_512);i++) W3[i] = tmp[i];
	karat_mult3_m512i( W2, W0, W4);


	//W4 = U2 * V2      ; W0 = U0 * V0
	karat_mult3_m512i( W4, U2, V2);
	karat_mult3_m512i( W0, U0, V0);

	
		
	//INTERPOLATION PHASE
	//9 add, 1 shift, 1 Smul, 2 Sdiv (2n)
	
	//W3 = W3 + W2
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W3[i] ^= W2[i];
		
	
	//W1 = W1 + W0
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W1[i] ^= W0[i];
		
	
	//W2 =(W2 + W0)/x -> x = X^64
	
	U1_64 = ((uint64_t *) W2)+1;
	U2_64 = ((uint64_t *) W0)+1;
	for(int i=0;i<(T_TM3_3W_512<<1);i++)
		{
			int i4 = i<<3;
			W2[i] = _mm512_loadu_si512((void const *)(& U1_64[i4]));
			W2[i] ^= _mm512_loadu_si512((void const *)(& U2_64[i4]));
		}
	
	

	//W2 =(W2 + W3 + W4*(x^3+1))/(x+1)
	
	U1_64 = ((uint64_t *) W4);
	__m512i * U1_512 = (__m512i *) (U1_64+5);
	
	tmp[0] = W2[0]^W3[0]^W4[0]^(__m512i){0x0ul,0x0ul,0x0ul,
						U1_64[0],U1_64[1],U1_64[2],U1_64[3],U1_64[4]};
	for(int i=1;i<(T_TM3_3W_512<<1);i++)
		tmp[i] = W2[i]^W3[i]^W4[i]^U1_512[i-1];

	
	divByXplus1_512(W2,tmp,T_TM3_3W_512);
	
	
	//W3 =(W3 + W1)/(x*(x+1))
	
	U1_64 = (uint64_t *) W3;
	U1_512 = (__m512i *) (U1_64+1);
	
	U2_64 = (uint64_t *) W1;
	__m512i * U2_512 = (__m512i *) (U2_64+1);
	
	for(int i=0;i<(T_TM3_3W_512<<1);i++)
		{tmp[i] = U1_512[i]^U2_512[i];}
		
	divByXplus1_512(W3,tmp,T_TM3_3W_512);
	const static __m512i mask = (const __m512i){0xffffffffffffffffUL,0xffffffffffffffffUL,
			0xffffffffffffffffUL,0xffffffffffffffffUL,
			0x0UL,0x0UL,0x0UL,0x0UL};
	W3[2*(T_TM3_3W_512)-1] &=mask;
	
	
	//W1 = W1 + W4 + W2
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W1[i] ^= W2[i]^W4[i];
	
	//W2 = W2 + W3
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W2[i] ^= W3[i];
	
	
	
	// Recomposition
	//W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4
	//Attention : W0, W1, W4 of size 2*T_TM3_3W_512, W2 and W3 of size 2*(T_TM3_3W_512)

	__m256i* ro256 = (__m256i*) ro512;
	__m256i* W0_256 = (__m256i*) W0;
	__m256i* W1_256 = (__m256i*) W1;
	__m256i* W2_256 = (__m256i*) W2;
	__m256i* W3_256 = (__m256i*) W3;
	__m256i* W4_256 = (__m256i*) W4;
	
	
	for(int i=0;i<(T_TM3_3W_256);i++)
	{
		ro512[i]=W0[i];
		_mm512_storeu_si512 ((__m512i*)(ro256 + ((T_TM3_3W_256+i)<<1)-1), W2[i]);
		ro512[i+(T_TM3_3W_256<<1)-1] = W4[i];
	}
	
	
	for(int i=(T_TM3_3W_256<<1);i<(T_TM3_3W_256<<1)-1;i++)
	{
		ro256[i]=W0_256[i];
		ro256[i+2*T_TM3_3W_256-1] = W2_256[i];
		ro256[i+4*T_TM3_3W_256-2] = W4_256[i];
	}
	
	ro256[(T_TM3_3W_256<<1)-1]=W0_256[(T_TM3_3W_256<<1)-1]^W2_256[0];
	ro256[(T_TM3_3W_256<<2)-2]=W2_256[(T_TM3_3W_256<<1)-1]^W4_256[0];
	ro256[(T_TM3_3W_256*6) -3]=W4_256[(T_TM3_3W_256<<1)-1];
	
	U1_64 = ((uint64_t *) &ro256[T_TM3_3W_256]);
	__m512i * T1_512 = (__m512i *) (U1_64-2);
	
	U2_64 = ((uint64_t *) &ro256[3*T_TM3_3W_256-1]);
	__m512i * T2_512 = (__m512i *) (U2_64-2);
	
	for(int i=0;i<T_TM3_3W_256;i++){
		__m512i t512 = _mm512_loadu_si512(T1_512+i);
		_mm512_storeu_si512(T1_512+i,t512^W1[i]);
		t512 = _mm512_loadu_si512(T2_512+i);
		_mm512_storeu_si512(T2_512+i,t512^W3[i]);
	}


	for(int i=0;i<3*T_TM3_3W_256-1;i++){
		_mm512_storeu_si512 ((Out)+i, ro512[i]);
	}

}

