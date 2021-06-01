#define WORD 64

#define T_3W_256 (T_3W >> 8)
#define T2_3W_256 (2 * T_3W_256)


inline static void karat_mult_1(__m128i *C, const __m128i *A, const __m128i *B);
inline static void karat_mult_2(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_4(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_8(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_16(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_32(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_64(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_128(__m256i *C, const __m256i *A, const __m256i *B);


inline static void karat_mult3(__m256i *C, const __m256i *A, const __m256i *B);



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

	// Computation of Al.Bl=D0 
	__m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);
	__m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);
	__m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));
	__m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e));
	__m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	//	Computation of Ah.Bh=D2
	DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);
	DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);
	AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));
	BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Computation of AlpAh.BlpBh=D1
	// initialisation of AlpAh and BlpBh
	__m128i AlpAh = _mm_xor_si128(Al, Ah);
	__m128i BlpBh = _mm_xor_si128(Bl, Bh);

	DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
	DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
	AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
	BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Computation of C
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
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_2(__m256i *C, const __m256i *A, const __m256i *B) {
	__m256i D0[2], D1[2], D2[2], SAA, SBB;
	__m128i *A128 = (__m128i *)A, *B128 = (__m128i *)B;

	karat_mult_1((__m128i *) D0, A128, B128);
	karat_mult_1((__m128i *) D2, A128 + 2, B128 + 2);
	SAA = A[0] ^ A[1];
	SBB = B[0] ^ B[1];
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
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_4(__m256i *C, const __m256i *A, const __m256i *B) {
	__m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];
			
	karat_mult_2(D0, A,B);
	karat_mult_2(D2, A + 2, B + 2);
	SAA[0] = A[0] ^ A[2];
	SBB[0] = B[0] ^ B[2];
	SAA[1] = A[1] ^ A[3];
	SBB[1] = B[1] ^ B[3];
	karat_mult_2(D1, SAA, SBB);
	
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
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_8(__m256i *C, const __m256i *A, const __m256i *B) {
	__m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];
			
	karat_mult_4(D0, A, B);
	karat_mult_4(D2, A + 4, B + 4);
	for(int32_t i = 0 ; i < 4 ; i++) {
		int32_t is = i + 4; 
		SAA[i] = A[i] ^ A[is];
		SBB[i] = B[i] ^ B[is];
	}

	karat_mult_4(D1, SAA, SBB);

	for(int32_t i = 0 ; i < 4 ; i++) {
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
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult_16(__m256i *C, const __m256i *A, const __m256i *B) {
	__m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];
			
	karat_mult_8(D0, A, B);
	karat_mult_8(D2, A + 8, B + 8);

	for(int32_t i = 0 ; i < 8 ; i++) {
		int32_t is = i + 8; 
		SAA[i] = A[i] ^ A[is];
		SBB[i] = B[i] ^ B[is];
	}

	karat_mult_8(D1, SAA, SBB);

	for(int32_t i = 0 ; i < 8 ; i++) {
		int32_t is = i + 8;
		int32_t is2 = is + 8;
		int32_t is3 = is2 + 8;
		
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
 * A(x) and B(x) are stored in 256-bit registers
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
 * This function computes A(x)*B(x) using Karatsuba 3 part split
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karat_mult3(__m256i *Out, const __m256i *A, const __m256i *B) {
	const __m256i *a0, *b0, *a1, *b1, *a2, *b2;
	__m256i aa01[T_3W_256], bb01[T_3W_256], aa02[T_3W_256], bb02[T_3W_256], aa12[T_3W_256], bb12[T_3W_256];
	__m256i D0[T2_3W_256], D1[T2_3W_256], D2[T2_3W_256], D3[T2_3W_256], D4[T2_3W_256], D5[T2_3W_256];
	__m256i ro256[3 * T2_3W_256];

	a0 = A;
	a1 = A + T_3W_256;
	a2 = A + (T_3W_256 << 1);

	b0 = B;
	b1 = B + T_3W_256;
	b2 = B + (T_3W_256 << 1);

	for (int32_t i = 0 ; i < T_3W_256 ; i++) {
		aa01[i] = a0[i] ^ a1[i];
		bb01[i] = b0[i] ^ b1[i];

		aa12[i] = a2[i] ^ a1[i];
		bb12[i] = b2[i] ^ b1[i];

		aa02[i] = a0[i] ^ a2[i];
		bb02[i] = b0[i] ^ b2[i];
	}


#if T_3W == 2048
	karat_mult_8(D0, a0, b0);
	karat_mult_8(D1, a1, b1);
	karat_mult_8(D2, a2, b2);

	karat_mult_8(D3, aa01, bb01);
	karat_mult_8(D4, aa02, bb02);
	karat_mult_8(D5, aa12, bb12);
#endif

#if T_3W == 4096
	karat_mult_16(D0, a0, b0);
	karat_mult_16(D1, a1, b1);
	karat_mult_16(D2, a2, b2);

	karat_mult_16(D3, aa01, bb01);
	karat_mult_16(D4, aa02, bb02);
	karat_mult_16(D5, aa12, bb12);
#endif

#if T_3W == 8192
	karat_mult_32(D0, a0, b0);
	karat_mult_32(D1, a1, b1);
	karat_mult_32(D2, a2, b2);

	karat_mult_32(D3, aa01, bb01);
	karat_mult_32(D4, aa02, bb02);
	karat_mult_32(D5, aa12, bb12);
#endif




	for (int32_t i = 0 ; i < T_3W_256 ; i++) {
		int32_t j = i + T_3W_256;
		__m256i middle0 = D0[i] ^ D1[i] ^ D0[j];
		ro256[i] = D0[i];
		ro256[j]  = D3[i] ^ middle0;
		ro256[j + T_3W_256] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
		middle0 = D1[j] ^ D2[i] ^ D2[j];
		ro256[j + (T_3W_256 << 1)] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
		ro256[i + (T_3W_256 << 2)] = D5[j] ^ middle0;
		ro256[j + (T_3W_256 << 2)] = D2[j];
	}

	for (int32_t i = 0 ; i < 3*T2_3W_256 ; i++) {
		Out[i] = ro256[i];
	}
}

