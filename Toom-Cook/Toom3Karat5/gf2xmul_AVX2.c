
#define WORD 64

#define T_5W_256 (T_5W >> 8)
#define T2_5W_256 (2 * T_5W_256)


inline static void karat_mult_1(__m128i *C, const __m128i *A, const __m128i *B);
inline static void karat_mult_2(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_4(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_8(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_16(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_32(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_64(__m256i *C, const __m256i *A, const __m256i *B);
inline static void karat_mult_128(__m256i *C, const __m256i *A, const __m256i *B);


inline static void karat_mult5(__m256i *C, const __m256i *A, const __m256i *B);



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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 256-bit registers
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
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_128(__m256i * C, const __m256i * A, const __m256i * B)
{
	__m256i D0[128],D1[128],D2[128],SAA[64],SBB[64];
	
	//printf("entrée dans karat128\n");
			
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
 * This function computes A(x)*B(x) using Karatsuba 5 part split
 * A(x) and B(x) are stored in 256-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult5(__m256i *Out, const __m256i *A, const __m256i *B) {
	const __m256i *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
	
	static __m256i aa01[T_5W_256], bb01[T_5W_256], aa02[T_5W_256], bb02[T_5W_256], aa03[T_5W_256], bb03[T_5W_256], aa04[T_5W_256], bb04[T_5W_256], 
			aa12[T_5W_256], bb12[T_5W_256], aa13[T_5W_256], bb13[T_5W_256], aa14[T_5W_256], bb14[T_5W_256],
			aa23[T_5W_256], bb23[T_5W_256], aa24[T_5W_256], bb24[T_5W_256],
			aa34[T_5W_256], bb34[T_5W_256];
	
	static __m256i D0[T2_5W_256], D1[T2_5W_256], D2[T2_5W_256], D3[T2_5W_256], D4[T2_5W_256], 
			D01[T2_5W_256], D02[T2_5W_256], D03[T2_5W_256], D04[T2_5W_256],
			D12[T2_5W_256], D13[T2_5W_256], D14[T2_5W_256],
			D23[T2_5W_256], D24[T2_5W_256],
			D34[T2_5W_256];

	__m256i ro256[t5 >> 1];

	a0 = A;
	a1 = a0 + T_5W_256;
	a2 = a1 + T_5W_256;
	a3 = a2 + T_5W_256;
	a4 = a3 + T_5W_256;
	b0 = B;
	b1 = b0 + T_5W_256;
	b2 = b1 + T_5W_256;
	b3 = b2 + T_5W_256;
	b4 = b3 + T_5W_256;

	for (int32_t i = 0 ; i < T_5W_256 ; i++)	{
		aa01[i] = a0[i] ^ a1[i];
		bb01[i] = b0[i] ^ b1[i];

		aa02[i] = a0[i] ^ a2[i];
		bb02[i] = b0[i] ^ b2[i];

		aa03[i] = a0[i] ^ a3[i];
		bb03[i] = b0[i] ^ b3[i];

		aa04[i] = a0[i] ^ a4[i];
		bb04[i] = b0[i] ^ b4[i];

		aa12[i] = a2[i] ^ a1[i];
		bb12[i] = b2[i] ^ b1[i];
		
		aa13[i] = a3[i] ^ a1[i];
		bb13[i] = b3[i] ^ b1[i];

		aa14[i] = a4[i] ^ a1[i];
		bb14[i] = b4[i] ^ b1[i];

		aa23[i] = a2[i] ^ a3[i];
		bb23[i] = b2[i] ^ b3[i];

		aa24[i] = a2[i] ^ a4[i];
		bb24[i] = b2[i] ^ b4[i];

		aa34[i] = a3[i] ^ a4[i];
		bb34[i] = b3[i] ^ b4[i];
	}
	
	
#if T_5W == 1024
	karat_mult_4(D0, a0, b0);
	karat_mult_4(D1, a1, b1);
	karat_mult_4(D2, a2, b2);
	karat_mult_4(D3, a3, b3);
	karat_mult_4(D4, a4, b4);

	karat_mult_4(D01, aa01, bb01);
	karat_mult_4(D02, aa02, bb02);
	karat_mult_4(D03, aa03, bb03);
	karat_mult_4(D04, aa04, bb04);
	
	karat_mult_4(D12, aa12, bb12);
	karat_mult_4(D13, aa13, bb13);
	karat_mult_4(D14, aa14, bb14);
	
	karat_mult_4(D23, aa23, bb23);
	karat_mult_4(D24, aa24, bb24);
	
	karat_mult_4(D34, aa34, bb34);
#endif
	
	
#if T_5W == 2048
	karat_mult_8(D0, a0, b0);
	karat_mult_8(D1, a1, b1);
	karat_mult_8(D2, a2, b2);
	karat_mult_8(D3, a3, b3);
	karat_mult_8(D4, a4, b4);

	karat_mult_8(D01, aa01, bb01);
	karat_mult_8(D02, aa02, bb02);
	karat_mult_8(D03, aa03, bb03);
	karat_mult_8(D04, aa04, bb04);
	
	karat_mult_8(D12, aa12, bb12);
	karat_mult_8(D13, aa13, bb13);
	karat_mult_8(D14, aa14, bb14);
	
	karat_mult_8(D23, aa23, bb23);
	karat_mult_8(D24, aa24, bb24);
	
	karat_mult_8(D34, aa34, bb34);
#endif
	
#if T_5W == 4096
	karat_mult_16(D0, a0, b0);
	karat_mult_16(D1, a1, b1);
	karat_mult_16(D2, a2, b2);
	karat_mult_16(D3, a3, b3);
	karat_mult_16(D4, a4, b4);
	
	karat_mult_16(D01, aa01, bb01);
	karat_mult_16(D02, aa02, bb02);
	karat_mult_16(D03, aa03, bb03);
	karat_mult_16(D04, aa04, bb04);
	
	karat_mult_16(D12, aa12, bb12);
	karat_mult_16(D13, aa13, bb13);
	karat_mult_16(D14, aa14, bb14);
	
	karat_mult_16(D23, aa23, bb23);
	karat_mult_16(D24, aa24, bb24);
	
	karat_mult_16(D34, aa34, bb34);
#endif

#if T_5W == 8192
	karat_mult_32(D0, a0, b0);
	karat_mult_32(D1, a1, b1);
	karat_mult_32(D2, a2, b2);
	karat_mult_32(D3, a3, b3);
	karat_mult_32(D4, a4, b4);
	
	karat_mult_32(D01, aa01, bb01);
	karat_mult_32(D02, aa02, bb02);
	karat_mult_32(D03, aa03, bb03);
	karat_mult_32(D04, aa04, bb04);
	
	karat_mult_32(D12, aa12, bb12);
	karat_mult_32(D13, aa13, bb13);
	karat_mult_32(D14, aa14, bb14);
	
	karat_mult_32(D23, aa23, bb23);
	karat_mult_32(D24, aa24, bb24);
	
	karat_mult_32(D34, aa34, bb34);
#endif



	
	for (int32_t i = 0 ; i < T_5W_256 ; i++) {
		ro256[i]            = D0[i];
		ro256[i + T_5W_256]   = D0[i + T_5W_256] ^ D01[i] ^ D0[i] ^ D1[i];
		ro256[i + 2 * T_5W_256] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[i + T_5W_256] ^ D0[i + T_5W_256] ^ D1[i + T_5W_256];
		ro256[i + 3 * T_5W_256] = D1[i + T_5W_256] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[i + T_5W_256] ^ D0[i + T_5W_256] ^ D2[i + T_5W_256];
		ro256[i + 4 * T_5W_256] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[i + T_5W_256] ^ D0[i + T_5W_256] ^ D3[i + T_5W_256] ^ D12[i + T_5W_256] ^ D1[i + T_5W_256] ^ D2[i + T_5W_256];
		ro256[i + 5 * T_5W_256] = D2[i + T_5W_256] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[i + T_5W_256] ^ D0[i + T_5W_256] ^ D4[i + T_5W_256] ^ D13[i + T_5W_256] ^ D1[i + T_5W_256] ^ D3[i + T_5W_256];
		ro256[i + 6 * T_5W_256] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[i + T_5W_256] ^ D1[i + T_5W_256] ^ D4[i + T_5W_256] ^ D23[i + T_5W_256] ^ D2[i + T_5W_256] ^ D3[i + T_5W_256];
		ro256[i + 7 * T_5W_256] = D3[i + T_5W_256] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[i + T_5W_256] ^ D2[i + T_5W_256] ^ D4[i + T_5W_256];
		ro256[i + 8 * T_5W_256] = D4[i] ^ D34[i + T_5W_256] ^ D3[i + T_5W_256] ^ D4[i + T_5W_256];
		ro256[i + 9 * T_5W_256] = D4[i + T_5W_256];
	}

	for(int32_t i = 0 ; i < T_5W_256 * 10 ; i++) {
		Out[i] = ro256[i];
	}
}



