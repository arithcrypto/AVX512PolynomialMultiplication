#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define WORD 64

#define T_3W_256 (T_3W>>8)
#define T2_3W_256 (2*T_3W_256)
#define TREC_3W_256 (SIZE_N>>8)
#define T2REC_3W_256 (2*TREC_3W_256)


inline static void karat_mult_1(__m128i *C, const __m128i *A, const __m128i *B);
inline static void karat_mult_2(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_4(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_8(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_16(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_32(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_64(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_128(__m256i *C, const __m256i *A, const __m256i *B);


/**
 * @brief Compute C(x) = A(x)*B(x)
 * A(x) and B(x) are stored in 128-bit registers
 * This function computes A(x)*B(x) using Karatsuba
 *
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karat_mult_1(__m128i *C, const __m128i *A, const __m128i *B) {
	__m128i D1[2];
	__m128i D0[2], D2[2];
	__m128i Al = _mm_loadu_si128(A);
	__m128i Ah = _mm_loadu_si128(A + 1);
	__m128i Bl = _mm_loadu_si128(B);
	__m128i Bh = _mm_loadu_si128(B + 1);

	//	Compute Al.Bl=D0
	__m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);
	__m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);
	__m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));
	__m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e));
	__m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	//	Compute Ah.Bh=D2
	DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);
	DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);
	AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));
	BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Compute AlpAh.BlpBh=D1
	// Initialisation of AlpAh and BlpBh
	__m128i AlpAh = _mm_xor_si128(Al,Ah);
	__m128i BlpBh = _mm_xor_si128(Bl,Bh);
	DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
	DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
	AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
	BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Final computation of C
	__m128i middle = _mm_xor_si128(D0[1], D2[0]);
	C[0] = D0[0];
	C[1] = middle ^ D0[0] ^ D1[0];
	C[2] = middle ^ D1[1] ^ D2[1];
	C[3] = D2[1];
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
inline static void karat_mult_2(__m256i *C, const __m256i *A, const __m256i *B) {
	__m256i D0[2], D1[2], D2[2], SAA, SBB;
	__m128i *A128 = (__m128i *)A, *B128 = (__m128i *)B;

	karat_mult_1((__m128i *) D0, A128, B128);
	karat_mult_1((__m128i *) D2, A128 + 2, B128 + 2);

	SAA = _mm256_xor_si256(A[0], A[1]);
	SBB = _mm256_xor_si256(B[0], B[1]);

	karat_mult_1((__m128i *) D1,(__m128i *) &SAA,(__m128i *) &SBB);
	__m256i middle = _mm256_xor_si256(D0[1], D2[0]);

	C[0] = D0[0];
	C[1] = middle ^ D0[0] ^ D1[0];
	C[2] = middle ^ D1[1] ^ D2[1];
	C[3] = D2[1];
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
inline static void karat_mult_4(__m256i *C, const __m256i *A, const __m256i *B) {
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
inline static void karat_mult_8(__m256i *C, const __m256i *A, const __m256i *B) {
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

inline static void karat_mult_16(__m256i * C, const __m256i * A, const __m256i * B)
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

inline static void karat_mult_32(__m256i * C, const __m256i * A, const __m256i * B)
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

inline static void karat_mult_64(__m256i * C, const __m256i * A, const __m256i * B)
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

inline static void karat_mult_128(__m256i * C, const __m256i * A, const __m256i * B)
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





