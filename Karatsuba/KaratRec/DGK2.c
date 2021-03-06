
#define PERM64(a , mask )            _mm512_permutex_epi64( a , mask )
#define PERM64X2(a , mask , b )      _mm512_permutex2var_epi64(a , mask , b )
#define PERM64VAR( mask , a )      _mm512_permutexvar_epi64( mask , a )
#define MUL(a , b , imm8 )      _mm512_clmulepi64_epi128(a , b , imm8 )
#define MXOR( src , mask , a , b )      _mm512_mask_xor_epi64( src , mask , a , b )
#define ALIGN(a , b , count )      _mm512_alignr_epi64(a , b , count )
#define STORE( mem , reg )      _mm512_storeu_si512( mem , reg )
#define LOAD( mem )      _mm512_loadu_si512( mem )
#define EXPANDLOAD( mask , mem )      _mm512_maskz_expandloadu_epi64( mask , mem )
#define LOAD128( mem )      _mm_loadu_si128( mem )
#define STORE128( mem , reg )      _mm_storeu_si128( mem , reg )
#define MUL128(a , b , imm8 )      _mm_clmulepi64_si128(a , b , imm8 )



static inline void mul128x4 ( __m512i *h , __m512i *l ,__m512i a , __m512i b )
{
	const __m512i mask_abq = _mm512_set_epi64 (6 , 7 , 4 , 5 , 2 , 3 , 0 , 1) ;
	__m512i s1 = a ^ PERM64(a , 0xb1 ) ; //(a , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	__m512i s2 = b ^ PERM64(b , 0xb1 ) ; //(b , _MM_SHUFFLE (2 , 3 , 0 , 1) ) ;
	__m512i lq = MUL (a , b , 0x00 ) ;
	__m512i hq = MUL (a , b , 0x11 ) ;
	__m512i abq = lq ^ hq ^ MUL ( s1 , s2 , 0x00 ) ;
	abq = PERM64VAR ( mask_abq , abq ) ;
	*l = MXOR ( lq , 0xaa , lq , abq ) ;
	*h = MXOR ( hq , 0x55 , hq , abq ) ;
}





static inline void karat_mult_1_512( __m512i *z ,const __m512i *a , const __m512i *b )
{
	const __m512i mask0 = _mm512_set_epi64 (13 , 12 , 5, 4, 9, 8 , 1 , 0) ;
	const __m512i mask1 = _mm512_set_epi64 (15 , 14 , 7, 6, 11, 10 , 3 , 2) ;
	const __m512i mask2 = _mm512_set_epi64 (3 , 2, 1, 0, 7, 6 , 5 , 4) ;
	const __m512i mask3 = _mm512_set_epi64 (11 , 10 , 9, 8, 3, 2 , 1 , 0) ;
	const __m512i mask4 = _mm512_set_epi64 (15 , 14 , 13, 12, 7, 6 , 5 , 4) ;
	const __m512i mask_s2 = _mm512_set_epi64 (3 , 2, 7, 6, 5, 4 , 1 , 0) ;
	const __m512i mask_s1 = _mm512_set_epi64 (7 , 6, 5, 4, 1, 0 , 3 , 2) ;


	__m512i t512[4], xh, xl, xab, xabh, xabl, xab1, xab2, yl, yh, yab, yabh, yabl;
	
	t512[0] = PERM64VAR ( mask_s1 , *a ) ^ PERM64VAR ( mask_s2 , *a ) ;
	t512[1] = PERM64VAR ( mask_s1 , *b ) ^ PERM64VAR ( mask_s2 , *b ) ;
	t512[2] = t512[0] ^ ALIGN ( t512[0] , t512[0] , 4) ;
	t512[3] = t512[1] ^ ALIGN ( t512[1] , t512[1] , 4) ;

	mul128x4 (& xh , & xl , *a , *b ) ;
	mul128x4 (& xabh , & xabl , t512[0] , t512[1]) ;
	mul128x4 (& yabh , & yabl , t512[2] , t512[3]) ;


	xab = xl ^ xh ^ PERM64X2 ( xabl , mask0 , xabh ) ;
	yl = PERM64X2 ( xl , mask3 , xh ) ;
	yh = PERM64X2 ( xl , mask4 , xh ) ;
	xab1 = ALIGN ( xab , xab , 6) ;
	xab2 = ALIGN ( xab , xab , 2) ;
	yl = MXOR ( yl , 0x3c , yl , xab1 ) ;
	yh = MXOR ( yh , 0x3c , yh , xab2 ) ;

	__m512i oxh= PERM64X2 ( xabl , mask1 , xabh ) ;
	__m512i oxl= ALIGN ( oxh , oxh , 4) ;
	yab= oxl ^ oxh ^ PERM64X2 ( yabl , mask0 , yabh ) ;
	yab= MXOR ( oxh , 0x3c , oxh , ALIGN ( yab , yab , 2) ) ;
	yab ^= yl ^yh ;

	yab = PERM64VAR ( mask2 , yab ) ;
	z[0] = MXOR ( yl , 0xf0 , yl , yab ) ;
	z[1] = MXOR ( yh , 0x0f , yh , yab ) ;

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
//	Karatsuba Multiplication
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


