#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define WORD 64
#define T_5W_512 (T_5W>>9)
#define T2_5W_512 (2*T_5W_512)
#define TREC_5W_512 (TREC_5W>>9)
#define T2REC_5W_512 (2*TREC_5W_512)

inline static void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_five_way_mult_m512i(__m512i *C, const __m512i *A, const __m512i *B);
inline static void karatRec_five_way_mult_m512i(__m512i *Out, const __m512i *A, const __m512i *B);



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
	/*
		Instruction count:
			- 13* _mm512_permutexvar_epi64
			- 3*  _mm512_permutex2var_epi64
			- 12* _mm512_clmulepi64_epi128
			- 11* _mm512_mask_xor_epi64
			- 11* XOR
			- 2* stores
	*/

	
	const __m512i perm_al = (__m512i){0x0UL,0x1UL,0x0UL,0x1UL,0x2UL,0x3UL,0x2UL,0x3UL};
	const __m512i perm_ah = (__m512i){0x4UL,0x5UL,0x4UL,0x5UL,0x6UL,0x7UL,0x6UL,0x7UL};
	const __m512i perm_bl = (__m512i){0x0UL,0x1UL,0x2UL,0x3UL,0x0UL,0x1UL,0x2UL,0x3UL};
	const __m512i perm_bh = (__m512i){0x4UL,0x5UL,0x6UL,0x7UL,0x4UL,0x5UL,0x6UL,0x7UL};
	const __m512i mask_R1 = _mm512_set_epi64 (6 , 7 , 4 , 5 , 2 , 3 , 0 , 1) ;
	const __m512i perm_h = (__m512i){0x4UL,0x5UL,0x0UL,0x1UL,0x2UL,0x3UL,0x6UL,0x7UL};
	const __m512i perm_l = (__m512i){0x0UL,0x1UL,0x4UL,0x5UL,0x6UL,0x7UL,0x2UL,0x3UL};
	const __m512i mask = _mm512_set_epi64 (15,14,13,12,3,2,1,0);
	
	__m512i al = _mm512_permutexvar_epi64(perm_al, *A );
	__m512i ah = _mm512_permutexvar_epi64(perm_ah, *A );
	__m512i bl = _mm512_permutexvar_epi64(perm_bl, *B );
	__m512i bh = _mm512_permutexvar_epi64(perm_bh, *B );
	
	__m512i sa = al^ah;
	__m512i sb = bl^bh;
	
	
	// First schoolbook multiplication 256 : AlBl
	
	__m512i R0_512=_mm512_clmulepi64_epi128(al,bl,0x00);
	__m512i R1_512=_mm512_clmulepi64_epi128(al,bl,0x01);
	__m512i R2_512=_mm512_clmulepi64_epi128(al,bl,0x10);
	__m512i R3_512=_mm512_clmulepi64_epi128(al,bl,0x11);

	R1_512 = _mm512_permutexvar_epi64( mask_R1 , R1_512^R2_512 ) ;

	__m512i l =  _mm512_mask_xor_epi64( R0_512 , 0xaa , R0_512 , R1_512 ) ;
	__m512i h =  _mm512_mask_xor_epi64( R3_512 , 0x55 , R3_512 , R1_512 ) ;
	
	__m512i cl = _mm512_permutex2var_epi64(l,mask,h);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	__m512i middle = _mm512_maskz_xor_epi64(0x3c,h,l);

	cl ^= middle;
	
	
	
	// Second schoolbook multiplication 256 : AhBh
	
	R0_512=_mm512_clmulepi64_epi128(ah,bh,0x00);
	R1_512=_mm512_clmulepi64_epi128(ah,bh,0x01);
	R2_512=_mm512_clmulepi64_epi128(ah,bh,0x10);
	R3_512=_mm512_clmulepi64_epi128(ah,bh,0x11);

	R1_512 = _mm512_permutexvar_epi64( mask_R1 , R1_512^R2_512 ) ;

	l =  _mm512_mask_xor_epi64( R0_512 , 0xaa , R0_512 , R1_512 ) ;
	h =  _mm512_mask_xor_epi64( R3_512 , 0x55 , R3_512 , R1_512 ) ;

	
	__m512i ch = _mm512_permutex2var_epi64(l,mask,h);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	middle = _mm512_maskz_xor_epi64(0x3c,h,l);

	ch ^= middle;
	
	
	// Third schoolbook multiplication 256 : SASB
	
	R0_512=_mm512_clmulepi64_epi128(sa,sb,0x00);
	R1_512=_mm512_clmulepi64_epi128(sa,sb,0x01);
	R2_512=_mm512_clmulepi64_epi128(sa,sb,0x10);
	R3_512=_mm512_clmulepi64_epi128(sa,sb,0x11);

	R1_512 = _mm512_permutexvar_epi64( mask_R1 , R1_512^R2_512 ) ;

	l =  _mm512_mask_xor_epi64( R0_512 , 0xaa , R0_512 , R1_512 ) ;
	h =  _mm512_mask_xor_epi64( R3_512 , 0x55 , R3_512 , R1_512 ) ;
	
	__m512i cm = _mm512_permutex2var_epi64(l,mask,h);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	middle = _mm512_maskz_xor_epi64(0x3c,h,l);

	cm ^= middle^cl^ch;


	// Final Reconstruction
	
	const __m512i perm_cm = (__m512i){0x4UL,0x5UL,0x6UL,0x7UL,0x0UL,0x1UL,0x2UL,0x3UL};
	cm = _mm512_permutexvar_epi64(perm_cm, cm );	
	
	C[0]= _mm512_mask_xor_epi64(cl,0xf0,cl,cm);
	C[1]= _mm512_mask_xor_epi64(ch,0x0f,ch,cm);
	

}
	


/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 1023 are stored in two 512-bit words
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
 * A(x) and B(x) of degree at most 2047 are stored in four 512-bit words
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
 * A(x) and B(x) of degree at most 4095 are stored in eight 512-bit words
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
 * A(x) and B(x) are stored in (5 x T5_W/512) 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karat_five_way_mult_m512i(__m512i *Out, const __m512i *A, const __m512i *B) {
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
	
#if T_5W == 512
	karat_mult_1_512(D0, a0, b0);
	karat_mult_1_512(D1, a1, b1);
	karat_mult_1_512(D2, a2, b2);
	karat_mult_1_512(D3, a3, b3);
	karat_mult_1_512(D4, a4, b4);

	karat_mult_1_512(D01, aa01, bb01);
	karat_mult_1_512(D02, aa02, bb02);
	karat_mult_1_512(D03, aa03, bb03);
	karat_mult_1_512(D04, aa04, bb04);
	
	karat_mult_1_512(D12, aa12, bb12);
	karat_mult_1_512(D13, aa13, bb13);
	karat_mult_1_512(D14, aa14, bb14);

	karat_mult_1_512(D23, aa23, bb23);
	karat_mult_1_512(D24, aa24, bb24);
	
	karat_mult_1_512(D34, aa34, bb34);
	
#endif
	
	
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


#if T_5W == 16384
	karat_mult_32_512(D0, a0, b0);
	karat_mult_32_512(D1, a1, b1);
	karat_mult_32_512(D2, a2, b2);
	karat_mult_32_512(D3, a3, b3);
	karat_mult_32_512(D4, a4, b4);
	
	karat_mult_32_512(D01, aa01, bb01);
	karat_mult_32_512(D02, aa02, bb02);
	karat_mult_32_512(D03, aa03, bb03);
	karat_mult_32_512(D04, aa04, bb04);
	
	karat_mult_32_512(D12, aa12, bb12);
	karat_mult_32_512(D13, aa13, bb13);
	karat_mult_32_512(D14, aa14, bb14);
	
	karat_mult_32_512(D23, aa23, bb23);
	karat_mult_32_512(D24, aa24, bb24);
	
	karat_mult_32_512(D34, aa34, bb34);
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



/**
 * @brief Compute C(x) = A(x)*B(x) 
 *
 * This function computes A(x)*B(x) using Karatsuba 5 part split
 * A(x) and B(x) are stored  in (5 x TREC_3W/512) 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karatRec_five_way_mult_m512i(__m512i *Out, const __m512i *A, const __m512i *B) {
	const __m512i *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
	
	static __m512i aa01[TREC_5W_512], bb01[TREC_5W_512], aa02[TREC_5W_512], bb02[TREC_5W_512], aa03[TREC_5W_512], bb03[TREC_5W_512], aa04[TREC_5W_512], bb04[TREC_5W_512], 
			aa12[TREC_5W_512], bb12[TREC_5W_512], aa13[TREC_5W_512], bb13[TREC_5W_512], aa14[TREC_5W_512], bb14[TREC_5W_512],
			aa23[TREC_5W_512], bb23[TREC_5W_512], aa24[TREC_5W_512], bb24[TREC_5W_512],
			aa34[TREC_5W_512], bb34[TREC_5W_512];
	
	static __m512i D0[T2REC_5W_512], D1[T2REC_5W_512], D2[T2REC_5W_512], D3[T2REC_5W_512], D4[T2REC_5W_512], 
			D01[T2REC_5W_512], D02[T2REC_5W_512], D03[T2REC_5W_512], D04[T2REC_5W_512],
			D12[T2REC_5W_512], D13[T2REC_5W_512], D14[T2REC_5W_512],
			D23[T2REC_5W_512], D24[T2REC_5W_512],
			D34[T2REC_5W_512];

	__m512i ro512[t5REC >> 1];

	a0 = A;
	a1 = a0 + TREC_5W_512;
	a2 = a1 + TREC_5W_512;
	a3 = a2 + TREC_5W_512;
	a4 = a3 + TREC_5W_512;
	b0 = B;
	b1 = b0 + TREC_5W_512;
	b2 = b1 + TREC_5W_512;
	b3 = b2 + TREC_5W_512;
	b4 = b3 + TREC_5W_512;

	for (int32_t i = 0 ; i < TREC_5W_512 ; i++)	{
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
	
	karat_five_way_mult_m512i(D0, a0, b0);
	karat_five_way_mult_m512i(D1, a1, b1);
	karat_five_way_mult_m512i(D2, a2, b2);
	karat_five_way_mult_m512i(D3, a3, b3);
	karat_five_way_mult_m512i(D4, a4, b4);

	karat_five_way_mult_m512i(D01, aa01, bb01);
	karat_five_way_mult_m512i(D02, aa02, bb02);
	karat_five_way_mult_m512i(D03, aa03, bb03);
	karat_five_way_mult_m512i(D04, aa04, bb04);
	
	karat_five_way_mult_m512i(D12, aa12, bb12);
	karat_five_way_mult_m512i(D13, aa13, bb13);
	karat_five_way_mult_m512i(D14, aa14, bb14);

	karat_five_way_mult_m512i(D23, aa23, bb23);
	karat_five_way_mult_m512i(D24, aa24, bb24);
	
	karat_five_way_mult_m512i(D34, aa34, bb34);




	for (int32_t i = 0 ; i < TREC_5W_512 ; i++) {
		ro512[i]            = D0[i];
		ro512[i + TREC_5W_512]   = D0[i + TREC_5W_512] ^ D01[i] ^ D0[i] ^ D1[i];
		ro512[i + 2 * TREC_5W_512] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[i + TREC_5W_512] ^ D0[i + TREC_5W_512] ^ D1[i + TREC_5W_512];
		ro512[i + 3 * TREC_5W_512] = D1[i + TREC_5W_512] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[i + TREC_5W_512] ^ D0[i + TREC_5W_512] ^ D2[i + TREC_5W_512];
		ro512[i + 4 * TREC_5W_512] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[i + TREC_5W_512] ^ D0[i + TREC_5W_512] ^ D3[i + TREC_5W_512] ^ D12[i + TREC_5W_512] ^ D1[i + TREC_5W_512] ^ D2[i + TREC_5W_512];
		ro512[i + 5 * TREC_5W_512] = D2[i + TREC_5W_512] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[i + TREC_5W_512] ^ D0[i + TREC_5W_512] ^ D4[i + TREC_5W_512] ^ D13[i + TREC_5W_512] ^ D1[i + TREC_5W_512] ^ D3[i + TREC_5W_512];
		ro512[i + 6 * TREC_5W_512] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[i + TREC_5W_512] ^ D1[i + TREC_5W_512] ^ D4[i + TREC_5W_512] ^ D23[i + TREC_5W_512] ^ D2[i + TREC_5W_512] ^ D3[i + TREC_5W_512];
		ro512[i + 7 * TREC_5W_512] = D3[i + TREC_5W_512] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[i + TREC_5W_512] ^ D2[i + TREC_5W_512] ^ D4[i + TREC_5W_512];
		ro512[i + 8 * TREC_5W_512] = D4[i] ^ D34[i + TREC_5W_512] ^ D3[i + TREC_5W_512] ^ D4[i + TREC_5W_512];
		ro512[i + 9 * TREC_5W_512] = D4[i + TREC_5W_512];
	}

	for(int32_t i = 0 ; i < TREC_5W_512 * 10 ; i++) {
		Out[i] = ro512[i];
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
int karat_mult5_5_AVX512(uint64_t *Out, const uint64_t *A, const uint64_t *B) {
	__m512i *A512 = (__m512i *)A, *B512 = (__m512i *)B,*C = (__m512i *)Out;

	karatRec_five_way_mult_m512i(C, A512, B512);

	
	return 0;
}






