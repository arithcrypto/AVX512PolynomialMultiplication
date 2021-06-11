/**
 * \file gf2x.c
 * \brief AVX2 implementation of multiplication of two polynomials
 */

#include "gf2x.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64
#define LAST64 (PARAM_N >> 6)

#define T_3W 4096
#define T_3W_512 (T_3W>>9)
#define T2_3W_512 (2*T_3W_512)
//#define TREC_3W 6144
//#define TREC_3W_256 (TREC_3W>>8)
#define T2REC_3W_512 (6*T_3W_512)

#define T_TM3_3W (PARAM_N_MULT / 3)
#define T_TM3 (PARAM_N_MULT + 384)
#define T_TM3_3W_512 ((T_TM3_3W + 128) / (8 * WORD))
#define T_TM3_3W_256 ((T_TM3_3W + 128) / (4 * WORD))
#define T_TM3_3W_64 (T_TM3_3W_256 << 2)


union int512_t{
	uint64_t i64[8];
	__m128i i128[4];
	__m256i i256[2];
	__m512i i512[1];

};

typedef union int512_t int512;

__m512i a1_times_a2[VEC_N_512_SIZE_64>>2];
__m512i o512[VEC_N_512_SIZE_64>>3];
uint64_t *tmp_reduce = (uint64_t *) o512;

uint64_t bloc64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
uint64_t bit64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight

static inline void reduce(__m256i *o, const __m256i *a);

static inline void karat_mult_256_256_512(__m512i * C, const __m256i * A, const __m256i * B);
static inline void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult3_m512i(__m512i *C, const __m512i *A, const __m512i *B);


static inline void divide_by_x_plus_one_512(__m512i *out, const __m512i *in, int32_t size);
static inline void toom_3_mult_512(__m512i * Out, const __m512i * A512, const __m512i * B512);


/**
 * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
 *
 * This function computes the modular reduction of the polynomial a(x)
 *
 * @param[out] o Pointer to the result
 * @param[in] a Pointer to the polynomial a(x)
 */
static inline void reduce(__m256i *o, const __m256i *a256) {
    __m256i r256, carry256;
    uint64_t *a = (uint64_t *) a256;
    static const int32_t dec64 = PARAM_N & 0x3f;
    static int32_t d0;
    int32_t i, i2;
	__m256i * o256 = (__m256i*) o512;
    d0 = WORD - dec64;
    for (i = LAST64 ; i < (PARAM_N >> 5) - 4 ; i += 4) {
        r256 = _mm256_lddqu_si256((__m256i const *) (& a[i]));
        r256 = _mm256_srli_epi64(r256, dec64);
        carry256 = _mm256_lddqu_si256((__m256i const *) (& a[i + 1]));
        carry256 = _mm256_slli_epi64(carry256, d0);
        r256 ^= carry256;
        i2 = (i - LAST64) >> 2;
        o256[i2] = a256[i2] ^ r256;
    }

    i = i - LAST64;

    for (; i < LAST64 + 1 ; i++) {
        uint64_t r = a[i + LAST64] >> dec64;
        uint64_t carry = a[i + LAST64 + 1] << d0;
        r ^= carry;
        tmp_reduce[i] = a[i] ^ r;
    }

    tmp_reduce[LAST64] &= RED_MASK;
    memcpy(o, tmp_reduce, VEC_N_SIZE_BYTES);
}


/**
 * @brief Compute C(x) = A(x)*B(x) 
 * A(x) and B(x) are stored in 512-bit registers
 * This function computes A(x)*B(x) using Karatsuba
 *
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */


__m512i mask_middle= (__m512i){0x0UL,0xffffffffffffffffUL,0xffffffffffffffffUL,0x0UL,0x0UL,0xffffffffffffffffUL,0xffffffffffffffffUL,0x0UL};

//__m512i idx_a=(__m512i){0x0UL,0x2UL,0x1UL,0x3UL,0x0UL,0x2UL,0x1UL,0x3UL};

__m512i idx_b=(__m512i){0x0UL,0x1UL,0x2UL,0x3UL,0x2UL,0x3UL,0x0UL,0x1UL};
__m512i idx_1=(__m512i){0x0UL,0x1UL,0x8UL,0x9UL,0x2UL,0x3UL,0xaUL,0xbUL};
__m512i idx_2=(__m512i){0x0UL,0x1UL,0x6UL,0x7UL,0x2UL,0x3UL,0x4UL,0x5UL};
__m512i idx_3=(__m512i){0x0UL,0x1UL,0x4UL,0x5UL,0x2UL,0x3UL,0x6UL,0x7UL};
__m512i idx_4=(__m512i){0x8UL,0x0UL,0x1UL,0x2UL,0x3UL,0x4UL,0x5UL,0x8UL};
__m512i idx_5=(__m512i){0x8UL,0x8UL,0x8UL,0x6UL,0x7UL,0x8UL,0x8UL,0x8UL};
__m512i idx_6=(__m512i){0x0UL,0x0UL,0x4UL,0x5UL,0xcUL,0xdUL,0x0UL,0x0UL};
__m512i idx_7=(__m512i){0x0UL,0x0UL,0x6UL,0x7UL,0xeUL,0xfUL,0x0UL,0x0UL};


inline static void karat_mult_256_256_512(__m512i * Out, const __m256i * A256, const __m256i * B256)
{
	/*
		Complexité :
			- 2* _mm512_broadcast_i64x4
			- 3* _mm512_permutexvar_epi64
			- 5* _mm512_permutex2var_epi64
			- 4* clmulepi64_epi128
			- 5* XOR
	*/
	//printf("Pour Multiplication\n");
	
	//int512 A512;
	__m512i A512, B512 ;
	
	__m512i R0_512,R1_512,R2_512, R3_512, middle, tmp;
	
	
	A512 =_mm512_broadcast_i64x4(*A256);
	tmp =_mm512_broadcast_i64x4(*B256);
	B512 =_mm512_permutexvar_epi64 (idx_b, tmp);
	
	/*printf("A512 = {a0,a1,a2,a3,a0,a1,a2,a3}\n");
	afficheVect((uint64_t *)&A512,"A512", 8);
	printf("B512 = {b0,b1,b2,b3,b2,b3,b0,b1}\n");
	afficheVect((uint64_t *)&B512,"B512", 8);//*/
	
	
	R0_512=_mm512_clmulepi64_epi128(A512,B512,0x00);
	R1_512=_mm512_clmulepi64_epi128(A512,B512,0x10);
	R2_512=_mm512_clmulepi64_epi128(A512,B512,0x01);
	R3_512=_mm512_clmulepi64_epi128(A512,B512,0x11);
	
	/*afficheVect((uint64_t*)&R0_512,"R0_512", 8);
	afficheVect((uint64_t*)&R1_512,"R1_512", 8);
	afficheVect((uint64_t*)&R2_512,"R2_512", 8);
	afficheVect((uint64_t*)&R3_512,"R3_512", 8);*/
	
	tmp =  _mm512_permutex2var_epi64 (R0_512, idx_1, R3_512);
	//afficheVect((uint64_t*)&tmp,"Out ", 8);
	
	middle = _mm512_permutexvar_epi64 (idx_2, R1_512);
	//afficheVect((uint64_t*)&middle,"middle", 8);
	middle ^=_mm512_permutexvar_epi64 (idx_3, R2_512);
	//afficheVect((uint64_t*)&middle,"middle", 8);
	
	//idx_b sert de 0_512
	tmp ^= _mm512_permutex2var_epi64 (middle, idx_4,idx_b);
	tmp ^= _mm512_permutex2var_epi64 (middle, idx_5,idx_b);
	//afficheVect((uint64_t*)&tmp,"Out ", 8);
	
	middle = _mm512_permutex2var_epi64 (R0_512, idx_6, R3_512) ^ _mm512_permutex2var_epi64 (R0_512, idx_7, R3_512);
	//afficheVect((uint64_t*)&middle,"middle", 8);
	
	*Out = tmp^middle;
	//afficheVect((uint64_t*)Out,"Out ", 8);
	
	
}







/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[1],D1[1],D2[1];
	__m256i SAA,SBB,
		//*D0_256 = (__m256i *) D0, *D1_256 = (__m256i *) D1, *D2_256 = (__m256i *) D2,
		*A_256 = (__m256i *) A, *B_256 = (__m256i *) B, *C_256 = (__m256i *) C;
	/*__m128i *A128 = (__m128i *)A, *B128 = (__m128i *)B;

	karat_mult_1((__m128i *) D0, A128, B128);
	karat_mult_1((__m128i *) D2, A128+2, B128+2);*/
	
	karat_mult_256_256_512( D0, A_256, B_256);
	karat_mult_256_256_512( D2, A_256+1, B_256+1);
	SAA=A_256[0]^A_256[1];SBB=B_256[0]^B_256[1];
	
	//karat_mult_1((__m128i *) D1,(__m128i *) &SAA,(__m128i *) &SBB);
	karat_mult_256_256_512( D1, &SAA, &SBB);
	
	/*__m256i middle = _mm256_xor_si256(D0_256[1], D2_256[0]);

	C_256[0] = D0_256[0];
	C_256[1] = middle^D0_256[0]^D1_256[0];
	C_256[2] = middle^D1_256[1]^D2_256[1];
	C_256[3] = D2_256[1];*/
	
	C[0]=D0[0];
	C[1]=D2[0];
	
	int512 middle;
	middle.i512[0] = D0[0]^D1[0]^D2[0];
	
	C_256[1] ^= middle.i256[0];
	C_256[2] ^= middle.i256[1];
	
	
	
}
	

/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[2],D1[2],D2[2];
	__m512i SAA,SBB;

	karat_mult_1_512( D0, A, B);
	karat_mult_1_512( D2, A+1, B+1);
	SAA=A[0]^A[1];SBB=B[0]^B[1];
	
	//karat_mult_1((__m128i *) D1,(__m128i *) &SAA,(__m128i *) &SBB);
	karat_mult_1_512( D1, &SAA, &SBB);
	
	__m512i middle = _mm512_xor_si512(D0[1], D2[0]);

	C[0] = D0[0];
	C[1] = middle^D0[0]^D1[0];
	C[2] = middle^D1[1]^D2[1];
	C[3] = D2[1];
}


/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[4],D1[4],D2[4],SAA[2],SBB[2];
			
	karat_mult_2_512( D0, A,B);
	karat_mult_2_512(D2,A+2,B+2);
	SAA[0]=A[0]^A[2];SBB[0]=B[0]^B[2];
	SAA[1]=A[1]^A[3];SBB[1]=B[1]^B[3];
	karat_mult_2_512( D1, SAA, SBB);
	
	__m512i middle0 = _mm512_xor_si512(D0[2], D2[0]);
	__m512i middle1 = _mm512_xor_si512(D0[3], D2[1]);
	
	C[0] = D0[0];
	C[1] = D0[1];
	C[2] = middle0^D0[0]^D1[0];
	C[3] = middle1^D0[1]^D1[1];
	C[4] = middle0^D1[2]^D2[2];
	C[5] = middle1^D1[3]^D2[3];
	C[6] = D2[2];
	C[7] = D2[3];
}
	
/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[8],D1[8],D2[8],SAA[4],SBB[4];
			
	karat_mult_4_512( D0, A,B);
	karat_mult_4_512(D2,A+4,B+4);
	for(int i=0;i<4;i++) {
		int is = i+4; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_4_512( D1, SAA, SBB);
	for(int i=0;i<4;i++)
	{
		int is = i+4;
		int is2 = is +4;
		int is3 = is2+4;
		
		__m512i middle = _mm512_xor_si512(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}



/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Karatsuba 3 part split
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karat_mult3_m512i(__m512i *Out, const __m512i *A, const __m512i *B) {
	const __m512i *a0, *b0, *a1, *b1, *a2, *b2;
	__m512i aa01[T_3W_512], bb01[T_3W_512], aa02[T_3W_512], bb02[T_3W_512], aa12[T_3W_512], bb12[T_3W_512];
	__m512i D0[T2_3W_512], D1[T2_3W_512], D2[T2_3W_512], D3[T2_3W_512], D4[T2_3W_512], D5[T2_3W_512];
	__m512i ro512[3 * T2_3W_512];

	a0 = A;
	a1 = A + T_3W_512;
	a2 = A + (T_3W_512 << 1);

	b0 = B;
	b1 = B + T_3W_512;
	b2 = B + (T_3W_512 << 1);

	for (int32_t i = 0 ; i < T_3W_512 ; i++) {
		aa01[i] = a0[i] ^ a1[i];
		bb01[i] = b0[i] ^ b1[i];

		aa12[i] = a2[i] ^ a1[i];
		bb12[i] = b2[i] ^ b1[i];

		aa02[i] = a0[i] ^ a2[i];
		bb02[i] = b0[i] ^ b2[i];
	}
	
	
	karat_mult_8_512(D0, a0, b0);
	karat_mult_8_512(D1, a1, b1);
	karat_mult_8_512(D2, a2, b2);

	karat_mult_8_512(D3, aa01, bb01);
	karat_mult_8_512(D4, aa02, bb02);
	karat_mult_8_512(D5, aa12, bb12);


	for (int32_t i = 0 ; i < T_3W_512 ; i++) {
		int32_t j = i + T_3W_512;
		__m512i middle0 = D0[i] ^ D1[i] ^ D0[j];
		ro512[i] = D0[i];
		ro512[j]  = D3[i] ^ middle0;
		ro512[j + T_3W_512] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
		middle0 = D1[j] ^ D2[i] ^ D2[j];
		ro512[j + (T_3W_512 << 1)] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
		ro512[i + (T_3W_512 << 2)] = D5[j] ^ middle0;
		ro512[j + (T_3W_512 << 2)] = D2[j];
	}

	for (int32_t i = 0 ; i < 3*T2_3W_512 ; i++) {
		Out[i] = ro512[i];
	}
}



/**
 * @brief Compute B(x) = A(x)/(x+1) 
 *
 * This function computes A(x)/(x+1) using a Quercia like algorithm
 * @param[out] out Pointer to the result
 * @param[in] in Pointer to the polynomial A(x)
 * @param[in] size used to define the number of coeeficients of A
 */


static inline void divide_by_x_plus_one_512(__m512i* out, const __m512i* in,int32_t size){
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

//int Toom3ult(uint64_t* Out, uint64_t* A,  uint64_t* B)
void toom_3_mult_512(__m512i * Out, const __m512i * A512, const __m512i * B512)
{
	static __m512i U0[T_TM3_3W_512], V0[T_TM3_3W_512], U1[T_TM3_3W_512], V1[T_TM3_3W_512], U2[T_TM3_3W_512], V2[T_TM3_3W_512];
	
	static __m512i W0[2*(T_TM3_3W_512)], W1[2*(T_TM3_3W_512)], W2[2*(T_TM3_3W_512)], W3[2*(T_TM3_3W_512)], W4[2*(T_TM3_3W_512)];
	static __m512i tmp[4*(T_TM3_3W_512)];// 
	
	static __m512i ro512[6*(T_TM3_3W_512)];
 
	//const __m512i zero = (__m512i){0ul,0ul,0ul,0ul,0ul,0ul,0ul,0ul};
	//printf("entrée dans Toom3Mult\n");
	
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
		//U1[i]= (__m512i){A[i4+T_TM3_3W_64-2],A[i41+T_TM3_3W_64-2],0x0ul,0x0ul};
		//V1[i]= (__m512i){B[i4+T_TM3_3W_64-2],B[i41+T_TM3_3W_64-2],0x0ul,0x0ul};
		
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

	}//*/
	

	/*for (int32_t i = 0 ; i < T_TM3_3W_256 ; i++) {
		U0[i]= A[i];
		V0[i]= B[i];
		U1[i]= A[i + T_TM3_3W_256];
		V1[i]= B[i + T_TM3_3W_256];
		U2[i]= A[i + T2];
		V2[i]= B[i + T2];
	}

	for (int32_t i = T_TM3_3W_256 ; i < T_TM3_3W_256 + 2 ; i++)	{
		U0[i] = zero;
		V0[i] = zero;
		U1[i] = zero;
		V1[i] = zero;
		U2[i] = zero;
		V2[i] = zero;
	}*/
	
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
	//printf("W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x !!!!!!!!\n");
	// décaler de 64 bits, x = X^64 !!!!!!!!!!!!!!!!!!!!!!!!!
	
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
		
	
	//W2 =(W2 + W0)/x -> x = X^64 // à corriger ???????????
	
	U1_64 = ((uint64_t *) W2)+1;
	U2_64 = ((uint64_t *) W0)+1;
	for(int i=0;i<(T_TM3_3W_512<<1);i++)
		{
			int i4 = i<<3;
			W2[i] = _mm512_loadu_si512((void const *)(& U1_64[i4]));
			W2[i] ^= _mm512_loadu_si512((void const *)(& U2_64[i4]));
		}
	
	
	static const __m512i mask = (const __m512i){0xffffffffffffffffUL,0xffffffffffffffffUL,
			0xffffffffffffffffUL,0xffffffffffffffffUL,
			0x0UL,0x0UL,0x0UL,0x0UL};

	//W2 =(W2 + W3 + W4*(x^3+1))/(x+1)
	
	U1_64 = ((uint64_t *) W4);
	__m512i * U1_512 = (__m512i *) (U1_64+5);
	
	tmp[0] = W2[0]^W3[0]^W4[0]^(__m512i){0x0ul,0x0ul,0x0ul,
						U1_64[0],U1_64[1],U1_64[2],U1_64[3],U1_64[4]};
	for(int i=1;i<(T_TM3_3W_512<<1);i++)
		tmp[i] = W2[i]^W3[i]^W4[i]^U1_512[i-1];

	//tmp[(T_TM3_3W_512<<1)-1] = W2[i]^W3[i]^W4[i]
	//							^(__m512i){	U1_64[0],U1_64[1],U1_64[2],U1_64[3],U1_64[4],
	//									0x0ul,0x0ul,0x0ul,0x0ul};
	//tmp[(T_TM3_3W_512<<1)-1] &=mask;
	
	divide_by_x_plus_one_512(W2,tmp,T_TM3_3W_512);
	//W2[2*(T_TM3_3W_256)-1] = zero;
	//W2[2*(T_TM3_3W_512)-1] &=mask;
	
	
	//W3 =(W3 + W1)/(x*(x+1))
	
	U1_64 = (uint64_t *) W3;
	U1_512 = (__m512i *) (U1_64+1);
	
	U2_64 = (uint64_t *) W1;
	__m512i * U2_512 = (__m512i *) (U2_64+1);
	
	for(int i=0;i<(T_TM3_3W_512<<1);i++)
		{tmp[i] = U1_512[i]^U2_512[i];}
		
	//tmp[(T_TM3_3W_512<<1)-1] &=mask;	
	divide_by_x_plus_one_512(W3,tmp,T_TM3_3W_512);
	//W3[2*(T_TM3_3W_256)-1] = zero;
	W3[2*(T_TM3_3W_512)-1] &=mask;
	
	
	//W1 = W1 + W4 + W2
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W1[i] ^= W2[i]^W4[i];
	
	//W2 = W2 + W3
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W2[i] ^= W3[i];
	//W1[2*(T_TM3_3W_512)-1] &=mask;
	//W2[2*(T_TM3_3W_512)-1] &=mask;
	
	
	
	// Recomposition
	//W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4
	//Attention : W0, W1, W4 of size 2*T_TM3_3W_512, W2 and W3 of size 2*(T_TM3_3W_512)

	__m256i* ro256 = (__m256i*) ro512;
	__m256i* W0_256 = (__m256i*) W0;
	//__m256i* W1_256 = (__m256i*) W1;
	__m256i* W2_256 = (__m256i*) W2;
	//__m256i* W3_256 = (__m256i*) W3;
	__m256i* W4_256 = (__m256i*) W4;
	
	
	for(int i=0;i<(T_TM3_3W_256);i++)
	{
		ro512[i]=W0[i];
		_mm512_storeu_si512 ((__m512i*)(ro256 + ((T_TM3_3W_256+i)<<1)-1), W2[i]);//ro256[i+2*T_TM3_3W_256-1] = W2_256[i];
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
		//uint64_t * out64 = Out+(i<<2);
		_mm512_storeu_si512 ((Out)+i, ro512[i]);
	}//*/
	
}



/**
 * @brief Multiply two polynomials modulo \f$ X^n - 1\f$.
 *
 * This functions multiplies a dense polynomial <b>a1</b> (of Hamming weight equal to <b>weight</b>)
 * and a dense polynomial <b>a2</b>. The multiplication is done modulo \f$ X^n - 1\f$.
 *
 * @param[out] o Pointer to the result
 * @param[in] a1 Pointer to a polynomial
 * @param[in] a2 Pointer to a polynomial
 */
void vect_mul(__m512i *o, const __m512i *a1, const __m512i *a2) {
    toom_3_mult_512(a1_times_a2, a1, a2);
    reduce((__m256i *) o, (__m256i *) a1_times_a2);

    // clear all
    #ifdef __STDC_LIB_EXT1__
        memset_s(a1_times_a2, 0, (VEC_N_512_SIZE_64>>2) * sizeof(__m512i));
    #else
        memset(a1_times_a2, 0, (VEC_N_512_SIZE_64>>2) * sizeof(__m512i));
    #endif
}
