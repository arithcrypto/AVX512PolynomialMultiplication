
#define WORD 64

#define T_TM3R_3W (SIZE_N / 3)
#define T_TM3R (SIZE_N + 384)
#define tTM3R ((T_TM3R) / WORD)
#define T_TM3R_3W_256 ((T_TM3R_3W + 128) / (4 * WORD))
#define T_TM3R_3W_64 (T_TM3R_3W_256 << 2)


#define T_5W_256 (T_5W >> 8)

#define T2_5W_256 (2 * T_5W_256)
#define t5 (5 * T_5W / WORD)

#include "gf2xmul_AVX2.c"

inline static void divide_by_x_plus_one_256(__m256i *in, __m256i *out, int32_t size);
void toom_3_mult(__m256i * Out, const __m256i * A256, const __m256i * B256);


/**
 * @brief Compute B(x) = A(x)/(x+1) 
 *
 * This function computes A(x)/(x+1) using a Quercia like algorithm
 * @param[out] out Pointer to the result
 * @param[in] in Pointer to the polynomial A(x)
 * @param[in] size used to define the number of coeficients of A
 */

static inline int divByXplus1(__m256i* out,__m256i* in,int size){//mod X^N !!! N = 
	uint64_t * A = (uint64_t *) in;
	uint64_t * B = (uint64_t *) out;
	
	B[0] = A[0];
	for(int i=1;i<2*(size<<2);i++)
		B[i]= B[i-1]^A[i];

	return 0;
}

/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Toom-Cook 3 part split
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

void toom_3_mult(__m256i * Out, const __m256i * A256, const __m256i * B256)
{
	static __m256i U0[T_TM3R_3W_256], V0[T_TM3R_3W_256], U1[T_TM3R_3W_256], V1[T_TM3R_3W_256], U2[T_TM3R_3W_256], V2[T_TM3R_3W_256];
	
	static __m256i W0[2*(T_TM3R_3W_256)], W1[2*(T_TM3R_3W_256)], W2[2*(T_TM3R_3W_256)], W3[2*(T_TM3R_3W_256)], W4[2*(T_TM3R_3W_256)];
	static __m256i tmp[4*(T_TM3R_3W_256)];// 
	
	static __m256i ro256[6*(T_TM3R_3W_256)];
 
	const __m256i zero = (__m256i){0ul,0ul,0ul,0ul};
	
	uint64_t *A = (uint64_t *)A256;
	uint64_t *B = (uint64_t *)B256;
	
	
	int T2 = T_TM3R_3W_64<<1;
	for(int i=0;i<T_TM3R_3W_256-1;i++)
	{
		
		int i4 = i<<2;
		int i42 = i4-2;
		U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
		V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
		U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i42+T_TM3R_3W_64]));
		V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i42+T_TM3R_3W_64]));
		U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4+T2-4]));
		V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4+T2-4]));
	}
	
	for(int i=T_TM3R_3W_256-1;i<T_TM3R_3W_256;i++)
	{
		int i4 = i<<2;
		int i41 = i4+1;
		
		U0[i]= (__m256i){A[i4],A[i41],0x0ul,0x0ul};
		V0[i]= (__m256i){B[i4],B[i41],0x0ul,0x0ul};
		
		U1[i]= (__m256i){A[i4+T_TM3R_3W_64-2],A[i41+T_TM3R_3W_64-2],0x0ul,0x0ul};
		V1[i]= (__m256i){B[i4+T_TM3R_3W_64-2],B[i41+T_TM3R_3W_64-2],0x0ul,0x0ul};
		
		U2[i]= (__m256i){A[i4-4+T2],A[i4-3+T2],0x0ul,0x0ul};
		V2[i]= (__m256i){B[i4-4+T2],B[i4-3+T2],0x0ul,0x0ul};

	}
	
	// EVALUATION PHASE : x= X^64
	// P(X): P0=(0); P1=(1); P2=(x); P3=(1+x); P4=(\infty)
	// Evaluation: 5*2 add, 2*2 shift; 5 mul (n)
	
	
	//W3 = U2 + U1 + U0 ; W2 = V2 + V1 + V0

	for(int i=0;i<T_TM3R_3W_256;i++)
	{
		W3[i]=U0[i]^U1[i]^U2[i];
		W2[i]=V0[i]^V1[i]^V2[i];
	}
	
	//W1 = W2 * W3
	
	karat_mult5(W1, W2, W3);
		
	//W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x (SIZE = T_TM3R_3W_256 !)
	
	uint64_t * U1_64 = ((uint64_t *) U1);
	uint64_t * U2_64 = ((uint64_t *) U2);
	
	uint64_t * V1_64 = ((uint64_t *) V1);
	uint64_t * V2_64 = ((uint64_t *) V2);
	
	W0[0] = (__m256i){0ul,U1_64[0],U1_64[1]^U2_64[0],U1_64[2]^U2_64[1]};
	W4[0] = (__m256i){0ul,V1_64[0],V1_64[1]^V2_64[0],V1_64[2]^V2_64[1]};
	
	U1_64 = ((uint64_t *) U1)-1;
	U2_64 = ((uint64_t *) U2)-2;
	
	V1_64 = ((uint64_t *) V1)-1;
	V2_64 = ((uint64_t *) V2)-2;
	
	for(int i=1;i<T_TM3R_3W_256;i++)
	{
		int i4 = i<<2;
		W0[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
		W0[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
		
		W4[i] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
		W4[i] ^= _mm256_lddqu_si256((__m256i const *)(& V2_64[i4]));
	}
	
	
	//W3 = W3 + W0      ; W2 = W2 + W4
	for(int i=0;i<T_TM3R_3W_256;i++)
	{
		W3[i] ^= W0[i];
		W2[i] ^= W4[i];
	}
	

	//W0 = W0 + U0      ; W4 = W4 + V0
	for(int i=0;i<T_TM3R_3W_256;i++)
	{
		W0[i] ^= U0[i];
		W4[i] ^= V0[i];
	}


	//W3 = W3 * W2      ; W2 = W0 * W4
	karat_mult5(tmp,  W3, W2);
	for(int i=0;i<2*(T_TM3R_3W_256);i++) W3[i] = tmp[i];
	karat_mult5(W2, W0, W4);
	

	//W4 = U2 * V2      ; W0 = U0 * V0
	karat_mult5(W4, U2, V2);
	karat_mult5(W0, U0, V0);


		
	//INTERPOLATION PHASE
	//9 add, 1 shift, 1 Smul, 2 Sdiv (2n)
	
	//W3 = W3 + W2
	for(int i=0;i<2*(T_TM3R_3W_256);i++)
		W3[i] ^= W2[i];
		
	
	//W1 = W1 + W0
	for(int i=0;i<2*(T_TM3R_3W_256);i++)
		W1[i] ^= W0[i];
		
	
	//W2 =(W2 + W0)/x -> x = X^64
	
	U1_64 = ((uint64_t *) W2)+1;
	U2_64 = ((uint64_t *) W0)+1;
	for(int i=0;i<(T_TM3R_3W_256<<1);i++)
		{
			int i4 = i<<2;
			W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
			W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
		}
	
	

	//W2 =(W2 + W3 + W4*(x^3+1))/(x+1)
	
	U1_64 = ((uint64_t *) W4);
	__m256i * U1_256 = (__m256i *) (U1_64+1);
	
	tmp[0] = W2[0]^W3[0]^W4[0]^(__m256i){0x0ul,0x0ul,0x0ul,U1_64[0]};
	for(int i=1;i<(T_TM3R_3W_256<<1)-1;i++)
		tmp[i] = W2[i]^W3[i]^W4[i]^U1_256[i-1];
		
	
	divByXplus1(W2,tmp,T_TM3R_3W_256);
	W2[2*(T_TM3R_3W_256)-1] = zero;
	
	//W3 =(W3 + W1)/(x*(x+1))
	
	U1_64 = (uint64_t *) W3;
	U1_256 = (__m256i *) (U1_64+1);
	
	U2_64 = (uint64_t *) W1;
	__m256i * U2_256 = (__m256i *) (U2_64+1);
	
	for(int i=0;i<(T_TM3R_3W_256<<1)-1;i++)
		{tmp[i] = U1_256[i]^U2_256[i];}
	
	divByXplus1(W3,tmp,T_TM3R_3W_256);
	W3[2*(T_TM3R_3W_256)-1] = zero;
	
	
	//W1 = W1 + W4 + W2
	for(int i=0;i<2*(T_TM3R_3W_256);i++)
		W1[i] ^= W2[i]^W4[i];
	
	//W2 = W2 + W3
	for(int i=0;i<2*(T_TM3R_3W_256);i++)
		W2[i] ^= W3[i];
	
	
	
	// Recomposition
	//W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4
	//Attention : W0, W1, W4 of size 2*T_TM3R_3W_256, W2 and W3 of size 2*(T_TM3R_3W_256)


	for(int i=0;i<(T_TM3R_3W_256<<1)-1;i++)
	{
		ro256[i]=W0[i];
		ro256[i+2*T_TM3R_3W_256-1] = W2[i];
		ro256[i+4*T_TM3R_3W_256-2] = W4[i];
	}
	
	ro256[(T_TM3R_3W_256<<1)-1]=W0[(T_TM3R_3W_256<<1)-1]^W2[0];
	ro256[(T_TM3R_3W_256<<2)-2]=W2[(T_TM3R_3W_256<<1)-1]^W4[0];
	ro256[(T_TM3R_3W_256*6) -3]=W4[(T_TM3R_3W_256<<1)-1];
	
	U1_64 = ((uint64_t *) &ro256[T_TM3R_3W_256]);
	U1_256 = (__m256i *) (U1_64-2);
	
	U2_64 = ((uint64_t *) &ro256[3*T_TM3R_3W_256-1]);
	U2_256 = (__m256i *) (U2_64-2);
	
	for(int i=0;i<T_TM3R_3W_256<<1;i++){
		U1_256[i]^=W1[i];
		U2_256[i]^=W3[i];
	}

	for(int i=0;i<6*T_TM3R_3W_256-2;i++){
		_mm256_storeu_si256 (Out+i, ro256[i]);
	}

}



