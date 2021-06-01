#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(SIZE_N, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64

inline static void DGK_mult_256_256_512(__m512i * C, const __m256i * A, const __m256i * B);
inline static void karat_mult_2(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_4(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_8(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_16(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_32(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_64(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_128(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_256(__m256i *C, __m256i *A, __m256i *B);
inline static void karat_mult_512(__m256i *C, __m256i *A, __m256i *B);





/**
 * @brief Compute C(x) = A(x)*B(x) 
 * A(x) and B(x) are stored in 256-bit words
 * This function computes A(x)*B(x) and implements the Drucker et al.'s approach
 * "Fast Multiplication of Binary Polynomials with the Forthcoming Vectorized VPCLMULQDQ Instruction"
 * in ARITH25
 *
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

__m256i idx_0=(__m256i){0x0UL,0x1UL,0x0UL,0x1UL};
__m256i idx_1=(__m256i){0x2UL,0x3UL,0x2,0x3UL};
__m256i idx_2=(__m256i){0x2UL,0x3UL,0x0UL,0x1UL};


inline static void DGK_mult_256_256_512(__m512i * Out, const __m256i * A256, const __m256i * B256)
{
	/*
		Instruction count:
			- 2* _mm256_align_i64x4
			- 6* _mm256_permutexvar_epi64
			- 8* clmulepi64_epi128
			- 15* XOR
	*/
	__m256i A0, A1, B, zero ;
	
	A0 = _mm256_lddqu_si256(A256);
	B = _mm256_lddqu_si256(B256);

	__m256i OddSumLow, OddSumHigh, EvenSumLow, EvenSumHigh;
	__m256i Even0, Even1, Even2, Even3;
	__m256i Odd0, Odd1, Odd2, Odd3;
	
	OddSumLow ^= OddSumLow;
	OddSumHigh ^= OddSumHigh;
	EvenSumLow ^= EvenSumLow;
	EvenSumHigh ^= EvenSumHigh;
	zero ^= zero;
	
	A1 =_mm256_permutexvar_epi64 (idx_1, *A256);
	A0 =_mm256_permutexvar_epi64 (idx_0, *A256);
	
	Even0 = _mm256_clmulepi64_epi128(A0,B,0x00);
	Even1 = _mm256_clmulepi64_epi128(A0,B,0x11);
	Even2 = _mm256_clmulepi64_epi128(A1,B,0x00);
	Even3 = _mm256_clmulepi64_epi128(A1,B,0x11);
	
	Odd0 = _mm256_clmulepi64_epi128(A0,B,0x10);
	Odd1 = _mm256_clmulepi64_epi128(A0,B,0x01);
	Odd2 = _mm256_clmulepi64_epi128(A1,B,0x10);
	Odd3 = _mm256_clmulepi64_epi128(A1,B,0x01);
	
	Even1 =_mm256_permutexvar_epi64 (idx_2, Even1);
	Even2 =_mm256_permutexvar_epi64 (idx_2, Even2);
	Odd2 =_mm256_permutexvar_epi64 (idx_2, Odd2);
	Odd3 =_mm256_permutexvar_epi64 (idx_2, Odd3);
	
	EvenSumLow = Even0;
	EvenSumLow = _mm256_mask_xor_epi64(EvenSumLow, 0xc,  Even1, EvenSumLow );
	EvenSumLow = _mm256_mask_xor_epi64(EvenSumLow, 0xc,  Even2, EvenSumLow );
	
	EvenSumHigh = Even3;
	EvenSumHigh = _mm256_mask_xor_epi64(EvenSumHigh, 0x3,  Even1, EvenSumHigh );
	EvenSumHigh = _mm256_mask_xor_epi64(EvenSumHigh, 0x3,  Even2, EvenSumHigh );
	
	OddSumLow = Odd1^Odd0;
	OddSumLow = _mm256_mask_xor_epi64(OddSumLow, 0xc,  Odd2, OddSumLow );
	OddSumLow = _mm256_mask_xor_epi64(OddSumLow, 0xc,  Odd3, OddSumLow );
	OddSumHigh = _mm256_mask_xor_epi64(OddSumHigh, 0x3,  Odd3, Odd2 );
	
	OddSumHigh = _mm256_alignr_epi64(OddSumHigh, OddSumLow, 0x3);
	OddSumLow = _mm256_alignr_epi64(OddSumLow,zero, 0x3);
	
	EvenSumLow ^= OddSumLow;
	EvenSumHigh ^= OddSumHigh;
	
	__m256i *Out_256 = (__m256i *) Out;
	
	_mm256_storeu_si256(Out_256, EvenSumLow);
	_mm256_storeu_si256(Out_256+1, EvenSumHigh);
	
}

/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 511 are stored in two 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_2(__m256i *C, __m256i *A, __m256i *B) {
	__m512i D0[1], D1[1], D2[1];
	
	__m256i SAA, SBB;

	DGK_mult_256_256_512( D0, A, B);
	DGK_mult_256_256_512( D2, A + 1, B + 1);


	SAA = _mm256_xor_si256(A[0], A[1]);
	SBB = _mm256_xor_si256(B[0], B[1]);

	DGK_mult_256_256_512(D1, &SAA, &SBB);
	
	__m256i * D0_256 = (__m256i *) D0, * D1_256 = (__m256i *) D1, * D2_256 = (__m256i *) D2;
	__m256i middle = _mm256_xor_si256(D0_256[1], D2_256[0]);

	C[0] = D0_256[0];
	C[1] = middle ^ D0_256[0] ^ D1_256[0];
	C[2] = middle ^ D1_256[1] ^ D2_256[1];
	C[3] = D2_256[1];
}



/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 1023 are stored in four 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_4(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];

	karat_mult_2(D0, A, B);
	karat_mult_2(D2, A + 2, B + 2);

	SAA[0] = A[0] ^ A[2];
	SBB[0] = B[0] ^ B[2];
	SAA[1] = A[1] ^ A[3];
	SBB[1] = B[1] ^ B[3];

	karat_mult_2( D1, SAA, SBB);

	__m256i middle0 = _mm256_xor_si256(D0[2], D2[0]);
	__m256i middle1 = _mm256_xor_si256(D0[3], D2[1]);

	C[0] = D0[0];
	C[1] = D0[1];
	C[2] = middle0 ^ D0[0] ^ D1[0];
	C[3] = middle1 ^ D0[1] ^ D1[1];
	C[4] = middle0 ^ D1[2] ^ D2[2];
	C[5] = middle1 ^ D1[3] ^ D2[3];
	C[6] = D2[2];
	C[7] = D2[3];
}



/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 2047 are stored in eight 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_8(__m256i *C, __m256i *A, __m256i *B) {
	__m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];

	karat_mult_4(D0, A, B);
	karat_mult_4(D2, A + 4, B + 4);

	for (int32_t i = 0 ; i < 4 ; i++) {
		int32_t is = i + 4;
		SAA[i] = A[i] ^ A[is];
		SBB[i] = B[i] ^ B[is];
	}

	karat_mult_4(D1, SAA, SBB);

	for (int32_t i = 0 ; i < 4 ; i++) {
		int32_t is = i + 4;
		int32_t is2 = is + 4;
		int32_t is3 = is2 + 4;

		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);

		C[i]   = D0[i];
		C[is]  = middle ^ D0[i] ^ D1[i];
		C[is2] = middle ^ D1[is] ^ D2[is];
		C[is3] = D2[is];
	}
}

/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 4095 are stored in sixteen 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_16(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[16],D1[16],D2[16],SAA[8],SBB[8];
			
	karat_mult_8( D0, A,B);
	karat_mult_8(D2,A+8,B+8);
	for(int i=0;i<8;i++) {
		int is = i+8; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_8( D1, SAA, SBB);
	for(int i=0;i<8;i++)
	{
		int is = i+8;
		int is2 = is +8;
		int is3 = is2+8;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}


/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 8191 are stored in 32 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_32(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[32],D1[32],D2[32],SAA[16],SBB[16];
			
	karat_mult_16( D0, A,B);
	karat_mult_16(D2,A+16,B+16);
	for(int i=0;i<16;i++) {
		int is = i+16; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_16( D1, SAA, SBB);
	for(int i=0;i<16;i++)
	{
		int is = i+16;
		int is2 = is +16;
		int is3 = is2+16;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}

/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 16383 are stored in 64 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_64(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[64],D1[64],D2[64],SAA[32],SBB[32];
			
	karat_mult_32( D0, A,B);
	karat_mult_32(D2,A+32,B+32);
	for(int i=0;i<32;i++) {
		int is = i+32; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_32( D1, SAA, SBB);
	for(int i=0;i<32;i++)
	{
		int is = i+32;
		int is2 = is +32;
		int is3 = is2+32;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}



/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 32767 are stored in 128 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_128(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[128],D1[128],D2[128],SAA[64],SBB[64];
			
	karat_mult_64( D0, A,B);
	karat_mult_64(D2,A+64,B+64);
	for(int i=0;i<64;i++) {
		int is = i+64; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_64( D1, SAA, SBB);
	for(int i=0;i<64;i++)
	{
		int is = i+64;
		int is2 = is +64;
		int is3 = is2+64;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}


/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 65535 are stored in 256 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_256(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[256],D1[256],D2[256],SAA[128],SBB[128];
			
	karat_mult_128( D0, A,B);
	karat_mult_128(D2,A+128,B+128);
	for(int i=0;i<128;i++) {
		int is = i+128; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_128( D1, SAA, SBB);
	for(int i=0;i<128;i++)
	{
		int is = i+128;
		int is2 = is +128;
		int is3 = is2+128;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}


/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 131071 are stored in 512 256-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_512(__m256i * C, __m256i * A, __m256i * B)
{
	__m256i D0[512],D1[512],D2[512],SAA[256],SBB[256];
			
	karat_mult_256( D0, A,B);
	karat_mult_256(D2,A+256,B+256);
	for(int i=0;i<256;i++) {
		int is = i+256; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_256( D1, SAA, SBB);
	for(int i=0;i<256;i++)
	{
		int is = i+256;
		int is2 = is +256;
		int is3 = is2+256;
		
		__m256i middle = _mm256_xor_si256(D0[is], D2[i]);
		
		C[i]   = D0[i];
		C[is]  = middle^D0[i]^D1[i];
		C[is2] = middle^D1[is]^D2[is];
		C[is3] = D2[is];
	}
}



/*************************************************************************************
//
//                       MULTIPLICATION
//
//	Wrapper 64 bits pour
//  Karatsuba récursif 128 bits avec PCLMULQDQ intrinsic
//                   SANS REDUCTION
//
*************************************************************************************/

inline int karatRec(uint64_t * C64, const uint64_t * A64, const uint64_t * B64, int size)//size est en nombre de mots de 64 bits !!!
{

	__m256i * A = (__m256i *) A64;
	__m256i * B = (__m256i *) B64;

	__m256i * C = (__m256i *) C64;
	
	size = size>>2;

	if(size == 1) DGK_mult_256_256_512((__m512i *)C, A, B);
	else if(size == 2) karat_mult_2(C, A, B);
	else if(size == 4) karat_mult_4(C, A, B);
	else if(size == 8) karat_mult_8(C, A, B);
	else if(size == 16) karat_mult_16(C, A, B);
	else if(size == 32) karat_mult_32(C, A, B);
	else if(size == 64) karat_mult_64(C, A, B);
	else if(size == 128) karat_mult_128(C, A, B);
	else if(size == 256) karat_mult_256(C, A, B);
	else if(size == 512) karat_mult_512(C, A, B);
	
	
	//KaratRecPclmul256(A,B,C,size>>2);
		

	return 0;

}


