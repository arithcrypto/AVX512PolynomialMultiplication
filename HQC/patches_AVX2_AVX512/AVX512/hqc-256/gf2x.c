/**
 * \file gf2x.c
 * \brief AVX2 implementation of multiplication of two polynomials
 */

#include "gf2x.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>


#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64
#define LAST64 (PARAM_N >> 6)

#define T_5W 4096
#define T_5W_512 (T_5W>>9)
#define T2_5W_512 (2*T_5W_512)
#define T2REC_3W_512 (6*T_3W_512)

#define T_TM3_3W (PARAM_N_MULT / 3)
#define T_TM3 (PARAM_N_MULT + 1024)
#define T_TM3_3W_512 ((T_TM3_3W + 1024) / (8 * WORD))
#define T_TM3_3W_256 ((T_TM3_3W + 1024) / (4 * WORD))
#define T_TM3_3W_64 (T_TM3_3W_256 << 2)


union int512_t{
	uint64_t i64[8];
	__m128i i128[4];
	__m256i i256[2];
	__m512i i512[1];

};

typedef union int512_t int512;

__m512i a1_times_a2[VEC_N_512_SIZE_64>>2];
__m512i o512[VEC_N_512_SIZE_64>>3];
uint64_t *tmp_reduce = (uint64_t *) o512;

uint64_t bloc64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight
uint64_t bit64[PARAM_OMEGA_R]; // Allocation with the biggest possible weight

static inline void reduce(__m256i *o, const __m256i *a);

static inline void karat_mult_1_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_2_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_4_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult_8_512(__m512i * C, const __m512i * A, const __m512i * B);
static inline void karat_mult5_m512i(__m512i *C, const __m512i *A, const __m512i *B);


static inline void divide_by_x_plus_one_512(__m512i *out, const __m512i *in, int32_t size);
static inline void toom_3_mult_512(__m512i * Out, const __m512i * A512, const __m512i * B512);


/**
 * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
 *
 * This function computes the modular reduction of the polynomial a(x)
 *
 * @param[out] o Pointer to the result
 * @param[in] a Pointer to the polynomial a(x)
 */
static inline void reduce(__m256i *o, const __m256i *a256) {
    __m256i r256, carry256;
    uint64_t *a = (uint64_t *) a256;
    static const int32_t dec64 = PARAM_N & 0x3f;
    static int32_t d0;
    int32_t i, i2;
	__m256i * o256 = (__m256i*) o512;
    d0 = WORD - dec64;
    for (i = LAST64 ; i < (PARAM_N >> 5) - 4 ; i += 4) {
        r256 = _mm256_lddqu_si256((__m256i const *) (& a[i]));
        r256 = _mm256_srli_epi64(r256, dec64);
        carry256 = _mm256_lddqu_si256((__m256i const *) (& a[i + 1]));
        carry256 = _mm256_slli_epi64(carry256, d0);
        r256 ^= carry256;
        i2 = (i - LAST64) >> 2;
        o256[i2] = a256[i2] ^ r256;
    }

    i = i - LAST64;

    for (; i < LAST64 + 1 ; i++) {
        uint64_t r = a[i + LAST64] >> dec64;
        uint64_t carry = a[i + LAST64 + 1] << d0;
        r ^= carry;
        tmp_reduce[i] = a[i] ^ r;
    }

    tmp_reduce[LAST64] &= RED_MASK;
    memcpy(o, tmp_reduce, VEC_N_SIZE_BYTES);
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

	__m512i ro512[T2_5W_512*5];

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
 * @brief Compute B(x) = A(x)/(x+1) 
 *
 * This function computes A(x)/(x+1) using a Quercia like algorithm
 * @param[out] out Pointer to the result
 * @param[in] in Pointer to the polynomial A(x)
 * @param[in] size used to define the number of coeeficients of A
 */


static inline void divide_by_x_plus_one_512(__m512i* out, const __m512i* in,int size){
	out[0] = in[0];	
	for(int32_t i = 1 ; i < 2 * (size + 2) ; i++) {
		out[i] = out[i - 1] ^ in[i];
	}

}


/**
 * @brief Compute C(x) = A(x)*B(x) using TOOM3Mult 
 *
 * This function computes A(x)*B(x) using TOOM-COOK3 Multiplication
 * @param[out] Out Pointer to the result
 * @param[in] A Pointer to the polynomial A(x)
 * @param[in] B Pointer to the polynomial B(x)
 */

//int Toom3ult(uint64_t* Out, uint64_t* A,  uint64_t* B)
void toom_3_mult_512(__m512i * Out, const __m512i * A, const __m512i * B)
{
	static __m512i U0[T_TM3_3W_512], V0[T_TM3_3W_512], U1[T_TM3_3W_512], V1[T_TM3_3W_512], U2[T_TM3_3W_512], V2[T_TM3_3W_512];
	
	static __m512i W0[2*(T_TM3_3W_512)], W1[2*(T_TM3_3W_512)], W2[2*(T_TM3_3W_512)], W3[2*(T_TM3_3W_512)], W4[2*(T_TM3_3W_512)];
	static __m512i tmp[2*(T_TM3_3W_512)+3];// 
	
	static __m512i ro512[6*(T_TM3_3W_512)];
 
	const __m512i zero = (__m512i){0ul,0ul,0ul,0ul,0ul,0ul,0ul,0ul};
	
	const int32_t tt32=T_TM3_3W_512 - 2;
	const int32_t T2 = tt32<<1;

	for (int32_t i = 0 ; i < T_TM3_3W_512-2 ; i++) {
		U0[i]= A[i];
		V0[i]= B[i];
		U1[i]= A[i + tt32];
		V1[i]= B[i + tt32];
		U2[i]= A[i + T2];
		V2[i]= B[i + T2];
	}

	for (int32_t i = tt32 ; i < T_TM3_3W_512 ; i++)	{
		U0[i] = zero;
		V0[i] = zero;
		U1[i] = zero;
		V1[i] = zero;
		U2[i] = zero;
		V2[i] = zero;
	}

	// EVALUATION PHASE : x= X^64
	// P(X): P0=(0); P1=(1); P2=(x); P3=(1+x); P4=(\infty)
	// Evaluation: 5*2 add, 2*2 shift; 5 mul (n)
	
	
	//W3 = U2 + U1 + U0 ; W2 = V2 + V1 + V0

	for(int i=0;i<T_TM3_3W_512-2;i++)
	{
		W3[i]=U0[i]^U1[i]^U2[i];
		W2[i]=V0[i]^V1[i]^V2[i];
	}
	W2[T_TM3_3W_512-2] = zero;
	W2[T_TM3_3W_512-1] = zero;
	W3[T_TM3_3W_512-2] = zero;
	W3[T_TM3_3W_512-1] = zero;
	
	//W1 = W2 * W3
	
	
	karat_mult5_m512i( W1, W2, W3);
	
	
	
	//W0 =(U1 + U2*x)*x ; W4 =(V1 + V2*x)*x (SIZE = T_TM3_3W_512 !)
	W0[0] = zero;
	W4[0] = zero;
	
	W0[1] = U1[0];
	W4[1] = V1[0];
	
	for (int32_t i = 1 ; i < T_TM3_3W_512 - 1 ; i++) {
		W0[i + 1] = U1[i] ^ U2[i - 1];
		W4[i + 1] = V1[i] ^ V2[i - 1];
	}

	W0[T_TM3_3W_512 - 1] = U2[T_TM3_3W_512 - 3];
	W4[T_TM3_3W_512 - 1] = V2[T_TM3_3W_512 - 3];
	
	

	
	//W3 = W3 + W0      ; W2 = W2 + W4
	for(int i=0;i<T_TM3_3W_512;i++)
	{
		W3[i] ^= W0[i];
		W2[i] ^= W4[i];
	}
	

	//W0 = W0 + U0      ; W4 = W4 + V0
	for(int i=0;i<T_TM3_3W_512;i++)
	{
		W0[i] ^= U0[i];
		W4[i] ^= V0[i];
	}


	//W3 = W3 * W2      ; W2 = W0 * W4
	karat_mult5_m512i( tmp, W3, W2);
	for(int i=0;i<2*(T_TM3_3W_512);i++) W3[i] = tmp[i];
	karat_mult5_m512i( W2, W0, W4);


	//W4 = U2 * V2      ; W0 = U0 * V0
	karat_mult5_m512i( W4, U2, V2);
	karat_mult5_m512i( W0, U0, V0);

	
		
	//INTERPOLATION PHASE
	//9 add, 1 shift, 1 Smul, 2 Sdiv (2n)
	
	//W3 = W3 + W2
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W3[i] ^= W2[i];
		
	
	//W1 = W1 + W0
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W1[i] ^= W0[i];
		
	
	//W2 =(W2 + W0)/x -> x = X^512
	for (int32_t i = 0 ; i < 2 * (T_TM3_3W_512) - 1 ; i++) {
		int32_t i1 = i + 1;
		W2[i] = W2[i1] ^ W0[i1];
	}

	W2[2 * (T_TM3_3W_512) - 1] = zero;
	
	
	
	//W2 =(W2 + W3 + W4*(x^3+1))/(x+1)
	for (int32_t i = 0 ; i < 2 * (T_TM3_3W_512) ; i++) {
		tmp[i] = W2[i] ^ W3[i] ^ W4[i];
	}

	tmp[2 * (T_TM3_3W_512)] = zero;
	tmp[2 * (T_TM3_3W_512) + 1] = zero;
	tmp[2 * (T_TM3_3W_512) + 2] = zero;

	for (int32_t i = 0 ; i < 2 * (tt32) ; i++) {
		tmp[i + 3] ^= W4[i];
	}	
	
	divide_by_x_plus_one_512(W2,tmp,T_TM3_3W_512-2);
	
	
	//W3 =(W3 + W1)/(x*(x+1))
	
	for (int32_t i = 0 ; i < 2 * (T_TM3_3W_512) - 1 ; i++) {
		int32_t i1 = i + 1;
		tmp[i] = W3[i1] ^ W1[i1];
	}

	tmp[2*(T_TM3_3W_512)-1] = zero;
	divide_by_x_plus_one_512(W3,tmp,T_TM3_3W_512-2);
	
	
	//W1 = W1 + W4 + W2
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W1[i] ^= W2[i]^W4[i];
	
	//W2 = W2 + W3
	for(int i=0;i<2*(T_TM3_3W_512);i++)
		W2[i] ^= W3[i];
	
	
	
	// Recomposition
	//W  = W0+ W1*x+ W2*x^2+ W3*x^3 + W4*x^4
	//Attention : W0, W1, W4 of size 2*T_TM3_3W_512, W2 and W3 of size 2*(T_TM3_3W_512)

	for (int32_t i = 0 ; i < T_TM3_3W_512-2 ; i++) {
		ro512[i] = W0[i];
		ro512[i + tt32] = W0[i + tt32] ^ W1[i];
		ro512[i + 2 * tt32] = W1[i + tt32] ^ W2[i];
		ro512[i + 3 * tt32] = W2[i + tt32] ^ W3[i];
		ro512[i + 4 * tt32] = W3[i + tt32] ^ W4[i];
		ro512[i + 5 * tt32] = W4[i + tt32];
	}

	ro512[4 * tt32] ^= W2[2 * tt32];
	ro512[5 * tt32] ^= W3[2 * tt32];

	ro512[1 + 4 * tt32] ^= W2[1 + T2];
	ro512[1 + 5 * tt32] ^= W3[1 + T2];

	ro512[2 + 4 * tt32] ^= W2[2 + T2];
	ro512[2 + 5 * tt32] ^= W3[2 + T2];

	ro512[3 + 4 * tt32] ^= W2[3 + T2];
	ro512[3 + 5 * tt32] ^= W3[3 + T2];


	for(int i=0;i<3*T_TM3_3W_256-12;i++){
		_mm512_storeu_si512 ((Out)+i, ro512[i]);
	}
	
}




/**
 * @brief Multiply two polynomials modulo \f$ X^n - 1\f$.
 *
 * This functions multiplies a dense polynomial <b>a1</b> (of Hamming weight equal to <b>weight</b>)
 * and a dense polynomial <b>a2</b>. The multiplication is done modulo \f$ X^n - 1\f$.
 *
 * @param[out] o Pointer to the result
 * @param[in] a1 Pointer to a polynomial
 * @param[in] a2 Pointer to a polynomial
 */
void vect_mul(__m512i *o, const __m512i *a1, const __m512i *a2) {
    toom_3_mult_512(a1_times_a2, a1, a2);
    reduce((__m256i *) o, (__m256i *) a1_times_a2);

    // clear all
    #ifdef __STDC_LIB_EXT1__
        memset_s(a1_times_a2, 0, (VEC_N_512_SIZE_64>>2) * sizeof(__m512i));
    #else
        memset(a1_times_a2, 0, (VEC_N_512_SIZE_64>>2) * sizeof(__m512i));
    #endif
}
