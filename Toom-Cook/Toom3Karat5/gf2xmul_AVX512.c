#define WORD 64


inline static void SB_mult_256_256_512(__m512i * C, const __m256i * A, const __m256i * B);
inline static void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult5_m512i(__m512i *C, const __m512i *A, const __m512i *B);


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


__m512i idx_b=(__m512i){0x0UL,0x1UL,0x2UL,0x3UL,0x2UL,0x3UL,0x0UL,0x1UL};
__m512i idx_1=(__m512i){0x0UL,0x1UL,0x8UL,0x9UL,0x2UL,0x3UL,0xaUL,0xbUL};
__m512i idx_2=(__m512i){0x0UL,0x1UL,0x6UL,0x7UL,0x2UL,0x3UL,0x4UL,0x5UL};
__m512i idx_3=(__m512i){0x0UL,0x1UL,0x4UL,0x5UL,0x2UL,0x3UL,0x6UL,0x7UL};
__m512i idx_4=(__m512i){0x8UL,0x0UL,0x1UL,0x2UL,0x3UL,0x4UL,0x5UL,0x8UL};
__m512i idx_5=(__m512i){0x8UL,0x8UL,0x8UL,0x6UL,0x7UL,0x8UL,0x8UL,0x8UL};
__m512i idx_6=(__m512i){0x0UL,0x0UL,0x4UL,0x5UL,0xcUL,0xdUL,0x0UL,0x0UL};
__m512i idx_7=(__m512i){0x0UL,0x0UL,0x6UL,0x7UL,0xeUL,0xfUL,0x0UL,0x0UL};


inline static void SB_mult_256_256_512(__m512i * Out, const __m256i * A256, const __m256i * B256)
{
	/*
		Complexity :
			- 2* _mm512_broadcast_i64x4
			- 3* _mm512_permutexvar_epi64
			- 5* _mm512_permutex2var_epi64
			- 4* clmulepi64_epi128
			- 5* XOR
	*/
	__m512i A512, B512 ;
	
	__m512i R0_512,R1_512,R2_512, R3_512, middle, tmp;
	
	
	A512 =_mm512_broadcast_i64x4(*A256);
	tmp =_mm512_broadcast_i64x4(*B256);
	B512 =_mm512_permutexvar_epi64 (idx_b, tmp);
	
	
	R0_512=_mm512_clmulepi64_epi128(A512,B512,0x00);
	R1_512=_mm512_clmulepi64_epi128(A512,B512,0x10);
	R2_512=_mm512_clmulepi64_epi128(A512,B512,0x01);
	R3_512=_mm512_clmulepi64_epi128(A512,B512,0x11);
	
	tmp =  _mm512_permutex2var_epi64 (R0_512, idx_1, R3_512);
	
	middle = _mm512_permutexvar_epi64 (idx_2, R1_512);
	middle ^=_mm512_permutexvar_epi64 (idx_3, R2_512);
	
	tmp ^= _mm512_permutex2var_epi64 (middle, idx_4,idx_b);
	tmp ^= _mm512_permutex2var_epi64 (middle, idx_5,idx_b);
	
	middle = _mm512_permutex2var_epi64 (R0_512, idx_6, R3_512) ^ _mm512_permutex2var_epi64 (R0_512, idx_7, R3_512);
	
	*Out = tmp^middle;
	
	
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
		*D0_256 = (__m256i *) D0, *D1_256 = (__m256i *) D1, *D2_256 = (__m256i *) D2,
		*A_256 = (__m256i *) A, *B_256 = (__m256i *) B, *C_256 = (__m256i *) C;
	
	SB_mult_256_256_512( D0, A_256, B_256);
	SB_mult_256_256_512( D2, A_256+1, B_256+1);
	SAA=A_256[0]^A_256[1];SBB=B_256[0]^B_256[1];
	
	SB_mult_256_256_512( D1, &SAA, &SBB);
	
	
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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_16_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[16],D1[16],D2[16],SAA[8],SBB[8];
			
	karat_mult_8_512( D0, A,B);
	karat_mult_8_512(D2,A+8,B+8);
	for(int i=0;i<8;i++) {
		int is = i+8; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_8_512( D1, SAA, SBB);
	for(int i=0;i<8;i++)
	{
		int is = i+8;
		int is2 = is +8;
		int is3 = is2+8;
		
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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */


inline static void karat_mult_32_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[32],D1[32],D2[32],SAA[16],SBB[16];
			
	karat_mult_16_512( D0, A,B);
	karat_mult_16_512(D2,A+16,B+16);
	for(int i=0;i<16;i++) {
		int is = i+16; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_16_512( D1, SAA, SBB);
	for(int i=0;i<16;i++)
	{
		int is = i+16;
		int is2 = is +16;
		int is3 = is2+16;
		
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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_64_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[64],D1[64],D2[64],SAA[32],SBB[32];
			
	karat_mult_32_512( D0, A,B);
	karat_mult_32_512(D2,A+32,B+32);
	for(int i=0;i<32;i++) {
		int is = i+32; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_32_512( D1, SAA, SBB);
	for(int i=0;i<32;i++)
	{
		int is = i+32;
		int is2 = is +32;
		int is3 = is2+32;
		
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
 * This function computes A(x)*B(x) using Karatsuba 5 part split
 * A(x) and B(x) are stored in 512-bit registers
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
 
inline static void karat_mult5_m512i(__m512i *Out, const __m512i *A, const __m512i *B) {
	const __m512i *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
	
	static __m512i aa01[T_5W_512], bb01[T_5W_512], aa02[T_5W_512], bb02[T_5W_512], aa03[T_5W_512], bb03[T_5W_512], aa04[T_5W_512], bb04[T_5W_512], 
			aa12[T_5W_512], bb12[T_5W_512], aa13[T_5W_512], bb13[T_5W_512], aa14[T_5W_512], bb14[T_5W_512],
			aa23[T_5W_512], bb23[T_5W_512], aa24[T_5W_512], bb24[T_5W_512],
			aa34[T_5W_512], bb34[T_5W_512];
	
	static __m512i D0[T2_5W_512], D1[T2_5W_512], D2[T2_5W_512], D3[T2_5W_512], D4[T2_5W_512], 
			D01[T2_5W_512], D02[T2_5W_512], D03[T2_5W_512], D04[T2_5W_512],
			D12[T2_5W_512], D13[T2_5W_512], D14[T2_5W_512],
			D23[T2_5W_512], D24[T2_5W_512],
			D34[T2_5W_512];

	__m512i ro512[t5 >> 1];

	a0 = A;
	a1 = a0 + T_5W_512;
	a2 = a1 + T_5W_512;
	a3 = a2 + T_5W_512;
	a4 = a3 + T_5W_512;
	b0 = B;
	b1 = b0 + T_5W_512;
	b2 = b1 + T_5W_512;
	b3 = b2 + T_5W_512;
	b4 = b3 + T_5W_512;

	for (int32_t i = 0 ; i < T_5W_512 ; i++)	{
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
	karat_mult_2_512(D0, a0, b0);
	karat_mult_2_512(D1, a1, b1);
	karat_mult_2_512(D2, a2, b2);
	karat_mult_2_512(D3, a3, b3);
	karat_mult_2_512(D4, a4, b4);

	karat_mult_2_512(D01, aa01, bb01);
	karat_mult_2_512(D02, aa02, bb02);
	karat_mult_2_512(D03, aa03, bb03);
	karat_mult_2_512(D04, aa04, bb04);
	
	karat_mult_2_512(D12, aa12, bb12);
	karat_mult_2_512(D13, aa13, bb13);
	karat_mult_2_512(D14, aa14, bb14);
	
	karat_mult_2_512(D23, aa23, bb23);
	karat_mult_2_512(D24, aa24, bb24);
	
	karat_mult_2_512(D34, aa34, bb34);
#endif
	
	
#if T_5W == 2048
	karat_mult_4_512(D0, a0, b0);
	karat_mult_4_512(D1, a1, b1);
	karat_mult_4_512(D2, a2, b2);
	karat_mult_4_512(D3, a3, b3);
	karat_mult_4_512(D4, a4, b4);

	karat_mult_4_512(D01, aa01, bb01);
	karat_mult_4_512(D02, aa02, bb02);
	karat_mult_4_512(D03, aa03, bb03);
	karat_mult_4_512(D04, aa04, bb04);
	
	karat_mult_4_512(D12, aa12, bb12);
	karat_mult_4_512(D13, aa13, bb13);
	karat_mult_4_512(D14, aa14, bb14);
	
	karat_mult_4_512(D23, aa23, bb23);
	karat_mult_4_512(D24, aa24, bb24);
	
	karat_mult_4_512(D34, aa34, bb34);
#endif
	
#if T_5W == 4096
	karat_mult_8_512(D0, a0, b0);
	karat_mult_8_512(D1, a1, b1);
	karat_mult_8_512(D2, a2, b2);
	karat_mult_8_512(D3, a3, b3);
	karat_mult_8_512(D4, a4, b4);

	karat_mult_8_512(D01, aa01, bb01);
	karat_mult_8_512(D02, aa02, bb02);
	karat_mult_8_512(D03, aa03, bb03);
	karat_mult_8_512(D04, aa04, bb04);
	
	karat_mult_8_512(D12, aa12, bb12);
	karat_mult_8_512(D13, aa13, bb13);
	karat_mult_8_512(D14, aa14, bb14);
	
	karat_mult_8_512(D23, aa23, bb23);
	karat_mult_8_512(D24, aa24, bb24);
	
	karat_mult_8_512(D34, aa34, bb34);
#endif

#if T_5W == 8192
	karat_mult_16_512(D0, a0, b0);
	karat_mult_16_512(D1, a1, b1);
	karat_mult_16_512(D2, a2, b2);
	karat_mult_16_512(D3, a3, b3);
	karat_mult_16_512(D4, a4, b4);
	
	karat_mult_16_512(D01, aa01, bb01);
	karat_mult_16_512(D02, aa02, bb02);
	karat_mult_16_512(D03, aa03, bb03);
	karat_mult_16_512(D04, aa04, bb04);
	
	karat_mult_16_512(D12, aa12, bb12);
	karat_mult_16_512(D13, aa13, bb13);
	karat_mult_16_512(D14, aa14, bb14);
	
	karat_mult_16_512(D23, aa23, bb23);
	karat_mult_16_512(D24, aa24, bb24);
	
	karat_mult_16_512(D34, aa34, bb34);
#endif



	for (int32_t i = 0 ; i < T_5W_512 ; i++) {
		ro512[i]            = D0[i];
		ro512[i + T_5W_512]   = D0[i + T_5W_512] ^ D01[i] ^ D0[i] ^ D1[i];
		ro512[i + 2 * T_5W_512] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[i + T_5W_512] ^ D0[i + T_5W_512] ^ D1[i + T_5W_512];
		ro512[i + 3 * T_5W_512] = D1[i + T_5W_512] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[i + T_5W_512] ^ D0[i + T_5W_512] ^ D2[i + T_5W_512];
		ro512[i + 4 * T_5W_512] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[i + T_5W_512] ^ D0[i + T_5W_512] ^ D3[i + T_5W_512] ^ D12[i + T_5W_512] ^ D1[i + T_5W_512] ^ D2[i + T_5W_512];
		ro512[i + 5 * T_5W_512] = D2[i + T_5W_512] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[i + T_5W_512] ^ D0[i + T_5W_512] ^ D4[i + T_5W_512] ^ D13[i + T_5W_512] ^ D1[i + T_5W_512] ^ D3[i + T_5W_512];
		ro512[i + 6 * T_5W_512] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[i + T_5W_512] ^ D1[i + T_5W_512] ^ D4[i + T_5W_512] ^ D23[i + T_5W_512] ^ D2[i + T_5W_512] ^ D3[i + T_5W_512];
		ro512[i + 7 * T_5W_512] = D3[i + T_5W_512] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[i + T_5W_512] ^ D2[i + T_5W_512] ^ D4[i + T_5W_512];
		ro512[i + 8 * T_5W_512] = D4[i] ^ D34[i + T_5W_512] ^ D3[i + T_5W_512] ^ D4[i + T_5W_512];
		ro512[i + 9 * T_5W_512] = D4[i + T_5W_512];
	}

	for(int32_t i = 0 ; i < T_5W_512 * 10 ; i++) {
		Out[i] = ro512[i];
	}
}



