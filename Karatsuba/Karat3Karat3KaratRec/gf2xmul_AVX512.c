#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define WORD 64
#define T_3W_512 (T_3W>>9)
#define T2_3W_512 (2*T_3W_512)
#define TREC_3W_512 (TREC_3W>>9)
#define T2REC_3W_512 (2*TREC_3W_512)

inline static void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_three_way_mult_512(__m512i *C, const __m512i *A, const __m512i *B);


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
 * A(x) and B(x) of degree at most 8191 are stored in sixteen 512-bit words
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
 * This function computes A(x)*B(x) using Karatsuba 3 part split
 * A(x) and B(x) are stored  in (3 x T3_W/512) 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
inline static void karat_three_way_mult_512(__m512i *Out, const __m512i *A, const __m512i *B) {
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
	
	
#if T_3W == 512
	karat_mult_1_512(D0, a0, b0);
	karat_mult_1_512(D1, a1, b1);
	karat_mult_1_512(D2, a2, b2);

	karat_mult_1_512(D3, aa01, bb01);
	karat_mult_1_512(D4, aa02, bb02);
	karat_mult_1_512(D5, aa12, bb12);
	
	
#endif
	
	
	
#if T_3W == 1024
	karat_mult_2_512(D0, a0, b0);
	karat_mult_2_512(D1, a1, b1);
	karat_mult_2_512(D2, a2, b2);

	karat_mult_2_512(D3, aa01, bb01);
	karat_mult_2_512(D4, aa02, bb02);
	karat_mult_2_512(D5, aa12, bb12);
#endif

#if T_3W == 2048
	karat_mult_4_512(D0, a0, b0);
	karat_mult_4_512(D1, a1, b1);
	karat_mult_4_512(D2, a2, b2);

	karat_mult_4_512(D3, aa01, bb01);
	karat_mult_4_512(D4, aa02, bb02);
	karat_mult_4_512(D5, aa12, bb12);
#endif

#if T_3W == 4096
	karat_mult_8_512(D0, a0, b0);
	karat_mult_8_512(D1, a1, b1);
	karat_mult_8_512(D2, a2, b2);

	karat_mult_8_512(D3, aa01, bb01);
	karat_mult_8_512(D4, aa02, bb02);
	karat_mult_8_512(D5, aa12, bb12);
#endif

#if T_3W == 8192
	karat_mult_16_512(D0, a0, b0);
	karat_mult_16_512(D1, a1, b1);
	karat_mult_16_512(D2, a2, b2);

	karat_mult_16_512(D3, aa01, bb01);
	karat_mult_16_512(D4, aa02, bb02);
	karat_mult_16_512(D5, aa12, bb12);
#endif




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

	for (int32_t i = 0 ; i < T2REC_3W_512 ; i++) {
		Out[i] = ro512[i];
	}
}



/**
 * @brief Compute C(x) = A(x)*B(x)
 *
 * This function computes A(x)*B(x) using Karatsuba 3 part split
 * A(x) and B(x) are stored  in (3 x TREC_3W/512) 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */
int karat_mult3_3_AVX512(uint64_t *Out, const uint64_t *A, const uint64_t *B) {
	__m512i *A512 = (__m512i *)A, *B512 = (__m512i *)B,*C = (__m512i *)Out;
	__m512i *a0, *b0, *a1, *b1, *a2, *b2;
	__m512i aa01[TREC_3W_512], bb01[TREC_3W_512], aa02[TREC_3W_512], bb02[TREC_3W_512], aa12[TREC_3W_512], bb12[TREC_3W_512];
	__m512i D0[T2REC_3W_512], D1[T2REC_3W_512], D2[T2REC_3W_512], D3[T2REC_3W_512], D4[T2REC_3W_512], D5[T2REC_3W_512];

	a0 = A512;
	a1 = A512+TREC_3W_512;
	a2 = A512+(T2REC_3W_512);

	b0 = B512;
	b1 = B512+TREC_3W_512;
	b2 = B512+(T2REC_3W_512);

	for (int32_t i = 0 ; i < TREC_3W_512 ; i++) {
		aa01[i] = a0[i] ^ a1[i];
		bb01[i] = b0[i] ^ b1[i];

		aa12[i] = a2[i] ^ a1[i];
		bb12[i] = b2[i] ^ b1[i];

		aa02[i] = a0[i] ^ a2[i];
		bb02[i] = b0[i] ^ b2[i];
	}

	karat_three_way_mult_512(D0, a0, b0);
	karat_three_way_mult_512(D1, a1, b1);
	karat_three_way_mult_512(D2, a2, b2);

	karat_three_way_mult_512(D3, aa01, bb01);
	karat_three_way_mult_512(D4, aa02, bb02);
	karat_three_way_mult_512(D5, aa12, bb12);

	for (int32_t i = 0 ; i < TREC_3W_512 ; i++) {
		int32_t j = i + TREC_3W_512;
		__m512i middle0 = D0[i] ^ D1[i] ^ D0[j];
		C[i] = D0[i];
		C[j]  = D3[i] ^ middle0;
		C[j + TREC_3W_512] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle0;
		middle0 = D1[j] ^ D2[i] ^ D2[j];
		C[j + (TREC_3W_512 << 1)] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle0;
		C[i + (TREC_3W_512 << 2)] = D5[j] ^ middle0;
		C[j + (TREC_3W_512 << 2)] = D2[j];
	}
	
	return 0;
}





