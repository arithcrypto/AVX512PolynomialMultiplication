#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>



inline static void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_16_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_32_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_64_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_128_512(__m512i * C, const __m512i * A, const __m512i * B);
inline static void karat_mult_256_512(__m512i * C, const __m512i * A, const __m512i * B);




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
	
	const __m512i perm_al = (__m512i){0x0UL,0x1UL,0x0UL,0x1UL,0x2UL,0x3UL,0x2UL,0x3UL};
	const __m512i perm_ah = (__m512i){0x4UL,0x5UL,0x4UL,0x5UL,0x6UL,0x7UL,0x6UL,0x7UL};
	
	const __m512i perm_bl = (__m512i){0x0UL,0x1UL,0x2UL,0x3UL,0x0UL,0x1UL,0x2UL,0x3UL};
	const __m512i perm_bh = (__m512i){0x4UL,0x5UL,0x6UL,0x7UL,0x4UL,0x5UL,0x6UL,0x7UL};
	
	const __m512i mask_D1 = _mm512_set_epi64 (6 , 7 , 4 , 5 , 2 , 3 , 0 , 1) ;
	
	const __m512i perm_h = (__m512i){0x4UL,0x5UL,0x0UL,0x1UL,0x2UL,0x3UL,0x6UL,0x7UL};

	const __m512i perm_l = (__m512i){0x0UL,0x1UL,0x4UL,0x5UL,0x6UL,0x7UL,0x2UL,0x3UL};
	
	const __m512i mask = _mm512_set_epi64 (15,14,13,12,3,2,1,0);
	
	__m512i al = _mm512_permutexvar_epi64(perm_al, *A );
	__m512i ah = _mm512_permutexvar_epi64(perm_ah, *A );
	__m512i bl = _mm512_permutexvar_epi64(perm_bl, *B );
	__m512i bh = _mm512_permutexvar_epi64(perm_bh, *B );
	
	__m512i sa = al^ah;
	__m512i sb = bl^bh;
	
	
	// Première multiplication 256 : AlBl
	
	__m512i s1 = al ^ _mm512_permutex_epi64(al , 0xb1 ) ; //(a , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	__m512i s2 = bl ^ _mm512_permutex_epi64(bl , 0xb1 ) ; //(b , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	__m512i D0_512=_mm512_clmulepi64_epi128(al,bl,0x00);
	__m512i D1_512=_mm512_clmulepi64_epi128(s1,s2,0x00);
	__m512i D2_512=_mm512_clmulepi64_epi128(al,bl,0x11);


	D1_512 = _mm512_permutexvar_epi64( mask_D1 , D0_512^D1_512^D2_512 ) ;
	__m512i l =  _mm512_mask_xor_epi64( D0_512 , 0xaa , D0_512 , D1_512 ) ;
	__m512i h =  _mm512_mask_xor_epi64( D2_512 , 0x55 , D2_512 , D1_512 ) ;
	
	__m512i cl = _mm512_permutex2var_epi64(l,mask,h);
	//afficheVect((uint64_t*)&c,"c", 8);
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	__m512i middle = _mm512_maskz_xor_epi64(0x3c,h,l);//*/
	//afficheVect((uint64_t*)&middle,"middle", 8);
	
	//afficheVect((uint64_t*)&c,"c", 8);
	cl ^= middle;
	
	
	
	// Deuxième multiplication 256 : AhBh
	
	s1 = ah ^ _mm512_permutex_epi64(ah , 0xb1 ) ; //(a , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	s2 = bh ^ _mm512_permutex_epi64(bh , 0xb1 ) ; //(b , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	D0_512=_mm512_clmulepi64_epi128(ah,bh,0x00);
	D1_512=_mm512_clmulepi64_epi128(s1,s2,0x00);
	D2_512=_mm512_clmulepi64_epi128(ah,bh,0x11);

	D1_512 = _mm512_permutexvar_epi64( mask_D1 , D0_512^D1_512^D2_512 ) ;
	l =  _mm512_mask_xor_epi64( D0_512 , 0xaa , D0_512 , D1_512 ) ;
	h =  _mm512_mask_xor_epi64( D2_512 , 0x55 , D2_512 , D1_512 ) ;

	
	__m512i ch = _mm512_permutex2var_epi64(l,mask,h);
	//afficheVect((uint64_t*)&c,"c", 8);
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	middle = _mm512_maskz_xor_epi64(0x3c,h,l);//*/
	//afficheVect((uint64_t*)&middle,"middle", 8);
	
	//afficheVect((uint64_t*)&c,"c", 8);
	ch ^= middle;
	
	
	// Troisième multiplication 256 : SASB
	
	s1 = sa ^ _mm512_permutex_epi64(sa , 0xb1 ) ; //(a , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	s2 = sb ^ _mm512_permutex_epi64(sb , 0xb1 ) ; //(b , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	D0_512=_mm512_clmulepi64_epi128(sa,sb,0x00);
	D1_512=_mm512_clmulepi64_epi128(s1,s2,0x00);
	D2_512=_mm512_clmulepi64_epi128(sa,sb,0x11);


	D1_512 = _mm512_permutexvar_epi64( mask_D1 , D0_512^D1_512^D2_512 ) ;
	l =  _mm512_mask_xor_epi64( D0_512 , 0xaa , D0_512 , D1_512 ) ;
	h =  _mm512_mask_xor_epi64( D2_512 , 0x55 , D2_512 , D1_512 ) ;
	
	__m512i cm = _mm512_permutex2var_epi64(l,mask,h);
	//afficheVect((uint64_t*)&c,"c", 8);
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	l = _mm512_permutexvar_epi64(perm_l, l );	
	h = _mm512_permutexvar_epi64(perm_h, h );
	
	//afficheVect((uint64_t*)&h,"h", 8);
	//afficheVect((uint64_t*)&l,"l", 8);
	middle = _mm512_maskz_xor_epi64(0x3c,h,l);//*/
	//afficheVect((uint64_t*)&middle,"middle", 8);
	
	//afficheVect((uint64_t*)&c,"c", 8);
	cm ^= middle^cl^ch;
	//afficheVect((uint64_t*)&h,"h", 8);
	const __m512i perm_cm = (__m512i){0x4UL,0x5UL,0x6UL,0x7UL,0x0UL,0x1UL,0x2UL,0x3UL};
	cm = _mm512_permutexvar_epi64(perm_cm, cm );	
	//afficheVect((uint64_t*)&h,"h", 8);
	
	C[0]= _mm512_mask_xor_epi64(cl,0xf0,cl,cm);
	C[1]= _mm512_mask_xor_epi64(ch,0x0f,ch,cm);
	
	/*int512 middle_512;
	__m256i *C_256 = (__m256i *) C;
	middle_512.i512[0] = cl^ch^cm;
	
	C_256[1] ^= middle_512.i256[0];
	C_256[2] ^= middle_512.i256[1];*/

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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 16383 are stored in 32 512-bit words
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
 * A(x) and B(x) of degree at most 32767 are stored in 64 512-bit words
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
 * This function computes A(x)*B(x) using Karatsuba
 * A(x) and B(x) of degree at most 65535 are stored in 128 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_128_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[128],D1[128],D2[128],SAA[64],SBB[64];
			
	karat_mult_64_512( D0, A,B);
	karat_mult_64_512(D2,A+64,B+64);
	for(int i=0;i<64;i++) {
		int is = i+64; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_64_512( D1, SAA, SBB);
	for(int i=0;i<64;i++)
	{
		int is = i+64;
		int is2 = is +64;
		int is3 = is2+64;
		
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
 * A(x) and B(x) of degree at most 131071 are stored in 256 512-bit words
 * @param[out] C Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

inline static void karat_mult_256_512(__m512i * C, const __m512i * A, const __m512i * B)
{
	__m512i D0[256],D1[256],D2[256],SAA[128],SBB[128];
			
	karat_mult_128_512( D0, A,B);
	karat_mult_128_512(D2,A+128,B+128);
	for(int i=0;i<128;i++) {
		int is = i+128; 
		SAA[i]=A[i]^A[is];SBB[i]=B[i]^B[is];
	}
	karat_mult_128_512( D1, SAA, SBB);
	for(int i=0;i<128;i++)
	{
		int is = i+128;
		int is2 = is +128;
		int is3 = is2+128;
		
		__m512i middle = _mm512_xor_si512(D0[is], D2[i]);
		
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
//	64 bit Wrapper for
//	Kartsuba Multiplication
//                   
//
*************************************************************************************/

inline int karatRec(uint64_t * C64, const uint64_t * A64, const uint64_t * B64, int size)//size is given as a number of 64-bit words !!!
{


	__m512i * A = (__m512i *) A64;
	__m512i * B = (__m512i *) B64;

	__m512i * C = (__m512i *) C64;
	
	size = size>>2;

	if(size == 4) karat_mult_2_512(C, A, B);
	else if(size == 8) karat_mult_4_512(C, A, B);
	else if(size == 16) karat_mult_8_512(C, A, B);
	else if(size == 32) karat_mult_16_512(C, A, B);
	else if(size == 64) karat_mult_32_512(C, A, B);
	else if(size == 128) karat_mult_64_512(C, A, B);
	else if(size == 256) karat_mult_128_512(C, A, B);
	else if(size == 512) karat_mult_256_512(C, A, B);

	

	return 0;

}


