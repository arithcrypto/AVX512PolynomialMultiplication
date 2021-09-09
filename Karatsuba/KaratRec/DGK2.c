
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





static inline void karat_mult_1_512( __m512i *z ,__m512i a , __m512i b )
{
	const __m512i mask0 = _mm512_set_epi64 (13 , 12 , 5, 4, 9, 8 , 1 , 0) ;
	const __m512i mask1 = _mm512_set_epi64 (15 , 14 , 7, 6, 11, 10 , 3 , 2) ;
	const __m512i mask2 = _mm512_set_epi64 (3 , 2, 1, 0, 7, 6 , 5 , 4) ;
	const __m512i mask3 = _mm512_set_epi64 (11 , 10 , 9, 8, 3, 2 , 1 , 0) ;
	const __m512i mask4 = _mm512_set_epi64 (15 , 14 , 13, 12, 7, 6 , 5 , 4) ;
	const __m512i mask_s2 = _mm512_set_epi64 (3 , 2, 7, 6, 5, 4 , 1 , 0) ;
	const __m512i mask_s1 = _mm512_set_epi64 (7 , 6, 5, 4, 1, 0 , 3 , 2) ;


	__m512i t512[4], xh, xl, xab, xabh, xabl, xab1, xab2, yl, yh, yab, yabh, yabl;
	
	t512[0] = PERM64VAR ( mask_s1 , a ) ^ PERM64VAR ( mask_s2 , a ) ;
	t512[1] = PERM64VAR ( mask_s1 , b ) ^ PERM64VAR ( mask_s2 , b ) ;
	t512[2] = t512[0] ^ ALIGN ( t512[0] , t512[0] , 4) ;
	t512[3] = t512[1] ^ ALIGN ( t512[1] , t512[1] , 4) ;

	mul128x4 (& xh , & xl , a , b ) ;
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


