/**
 * @file hqc.c
 * @brief Implementation of hqc.h
 */

#include "hqc.h"
#include "rng.h"
#include "parameters.h"
#include "parsing.h"
#include "gf2x.h"
#include "code.h"
#include "vector.h"
#include <stdint.h>
#include <string.h>
#ifdef VERBOSE
#include <stdio.h>
#endif


/**
 * @brief Keygen of the HQC_PKE IND_CPA scheme
 *
 * The public key is composed of the syndrome <b>s</b> as well as the <b>seed</b> used to generate the vector <b>h</b>.
 *
 * The secret key is composed of the <b>seed</b> used to generate vectors <b>x</b> and  <b>y</b>.
 * As a technicality, the public key is appended to the secret key in order to respect NIST API.
 *
 * @param[out] pk String containing the public key
 * @param[out] sk String containing the secret key
 */
void hqc_pke_keygen(unsigned char* pk, unsigned char* sk) {
	AES_XOF_struct sk_seedexpander;
	AES_XOF_struct pk_seedexpander;

    uint8_t sk_seed[SEED_BYTES] = {0};
    uint8_t pk_seed[SEED_BYTES] = {0};
    static __m512i h_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i y_512[VEC_N_512_SIZE_64 >> 3];   
    static __m512i x_512[VEC_N_512_SIZE_64 >> 3];
    static uint64_t s[VEC_N_256_SIZE_64];
    static __m512i tmp_512[VEC_N_512_SIZE_64 >> 3];

    #ifdef __STDC_LIB_EXT1__
        memset_s(x_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(y_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(h_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #else
        memset(x_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(y_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(h_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #endif

	// Create seed_expanders for public key and secret key
	randombytes(sk_seed, SEED_BYTES);
	seedexpander_init(&sk_seedexpander, sk_seed, sk_seed + 32, SEEDEXPANDER_MAX_LENGTH);

	randombytes(pk_seed, SEED_BYTES);
	seedexpander_init(&pk_seedexpander, pk_seed, pk_seed + 32, SEEDEXPANDER_MAX_LENGTH);

	// Compute secret key
    vect_set_random_fixed_weight(&sk_seedexpander, (__m256i *) x_512, PARAM_OMEGA);
    vect_set_random_fixed_weight(&sk_seedexpander, (__m256i *) y_512, PARAM_OMEGA);

	// Compute public key
    vect_set_random(&pk_seedexpander, (uint64_t *) h_512);
    vect_mul(tmp_512, y_512, h_512);
    vect_add(s, (uint64_t *) x_512, (uint64_t *) tmp_512, VEC_N_256_SIZE_64);

	// Parse keys to string
	hqc_public_key_to_string(pk, pk_seed, s);
	hqc_secret_key_to_string(sk, sk_seed, pk);

    #ifdef VERBOSE
        printf("\n\nsk_seed: "); for(int i = 0 ; i < SEED_BYTES ; ++i) printf("%02x", sk_seed[i]);
        printf("\n\nx: "); vect_print((uint64_t *) x_512, VEC_N_SIZE_BYTES);
        printf("\n\ny: "); vect_print((uint64_t *) y_512, VEC_N_SIZE_BYTES);

        printf("\n\npk_seed: "); for(int i = 0 ; i < SEED_BYTES ; ++i) printf("%02x", pk_seed[i]);
        printf("\n\nh: "); vect_print((uint64_t *) h_512, VEC_N_SIZE_BYTES);
        printf("\n\ns: "); vect_print(s, VEC_N_SIZE_BYTES);

        printf("\n\nsk: "); for(int i = 0 ; i < SECRET_KEY_BYTES ; ++i) printf("%02x", sk[i]);
        printf("\n\npk: "); for(int i = 0 ; i < PUBLIC_KEY_BYTES ; ++i) printf("%02x", pk[i]);
    #endif
}



/**
 * @brief Encryption of the HQC_PKE IND_CPA scheme
 *
 * The cihertext is composed of vectors <b>u</b> and <b>v</b>.
 *
 * @param[out] u Vector u (first part of the ciphertext)
 * @param[out] v Vector v (second part of the ciphertext)
 * @param[in] m Vector representing the message to encrypt
 * @param[in] theta Seed used to derive randomness required for encryption
 * @param[in] pk String containing the public key
 */
void hqc_pke_encrypt(uint64_t *u, uint64_t *v, uint64_t *m, unsigned char *theta, const unsigned char *pk) {
	AES_XOF_struct seedexpander;
    static __m512i h_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i s_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i r2_512[VEC_N_512_SIZE_64 >> 3];

    static __m512i r1_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i e_512[VEC_N_512_SIZE_64 >> 3];

    static __m512i tmp1_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i tmp2_512[VEC_N_512_SIZE_64 >> 3];
    static __m512i tmp3_512[VEC_N_512_SIZE_64 >> 3];
    static uint64_t tmp4[VEC_N_256_SIZE_64];

    #ifdef __STDC_LIB_EXT1__
        memset_s(r2_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(h_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(s_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(r1_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset_s(e_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #else
        memset(r2_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(h_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(s_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(r1_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
        memset(e_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #endif
    
	// Create seed_expander from theta
	seedexpander_init(&seedexpander, theta, theta + 32, SEEDEXPANDER_MAX_LENGTH);

	// Retrieve h and s from public key
    hqc_public_key_from_string((uint64_t *) h_512, (uint64_t *) s_512, pk);
	#ifdef VERBOSE
		printf("\n\n*s: "); vect_print((uint64_t *) s_512, VEC_N_SIZE_BYTES);
	#endif

	// Generate r1, r2 and e
    vect_set_random_fixed_weight(&seedexpander, (__m256i *) r1_512, PARAM_OMEGA_R);
    vect_set_random_fixed_weight(&seedexpander, (__m256i *) r2_512, PARAM_OMEGA_R);
    vect_set_random_fixed_weight(&seedexpander, (__m256i *) e_512, PARAM_OMEGA_E);
	

	// Compute u = r1 + r2.h
    vect_mul(tmp1_512, r2_512, h_512);
    vect_add(u, (uint64_t *) r1_512, (uint64_t *) tmp1_512, VEC_N_256_SIZE_64);

	// Compute v = m.G by encoding the message
    code_encode(v, m);
    vect_resize((uint64_t *) tmp2_512, PARAM_N, v, PARAM_N1N2);

	// Compute v = m.G + s.r2 + e
    vect_mul(tmp3_512, r2_512, s_512);
    vect_add(tmp4, (uint64_t *) e_512, (uint64_t *) tmp3_512, VEC_N_256_SIZE_64);
    vect_add((uint64_t *) tmp3_512, (uint64_t *) tmp2_512, tmp4, VEC_N_256_SIZE_64);
    vect_resize(v, PARAM_N1N2, (uint64_t *) tmp3_512, PARAM_N);

    #ifdef VERBOSE
        printf("\n\nh: "); vect_print((uint64_t *) h_512, VEC_N_SIZE_BYTES);
        printf("\n\ns: "); vect_print((uint64_t *) s_512, VEC_N_SIZE_BYTES);
        printf("\n\nr1: "); vect_print((uint64_t *) r1_512, VEC_N_SIZE_BYTES);
        printf("\n\nr2: "); vect_print((uint64_t *) r2_512, VEC_N_SIZE_BYTES);
        printf("\n\ne: "); vect_print((uint64_t *) e_512, VEC_N_SIZE_BYTES);
        printf("\n\ntmp3_256: "); vect_print((uint64_t *) tmp3_512, VEC_N_SIZE_BYTES);

        printf("\n\nu: "); vect_print(u, VEC_N_SIZE_BYTES);
        printf("\n\nv: "); vect_print(v, VEC_N1N2_SIZE_BYTES);
    #endif
}



/**
 * @brief Decryption of the HQC_PKE IND_CPA scheme
 *
 * @param[out] m Vector representing the decrypted message
 * @param[in] u Vector u (first part of the ciphertext)
 * @param[in] v Vector v (second part of the ciphertext)
 * @param[in] sk String containing the secret key
 */
void hqc_pke_decrypt(uint64_t *m, const __m512i *u_512, const uint64_t *v, const uint8_t *sk) {
    static __m512i x_512[VEC_N_512_SIZE_64 >> 3] = {0};
    static __m512i y_512[VEC_N_512_SIZE_64 >> 3] = {0};
    uint8_t pk[PUBLIC_KEY_BYTES] = {0};
    static uint64_t tmp1[VEC_N_256_SIZE_64] = {0};
    static uint64_t tmp2[VEC_N_256_SIZE_64] = {0};
    static __m512i tmp3_512[VEC_N_512_SIZE_64 >> 3];

    #ifdef __STDC_LIB_EXT1__
        memset_s(y_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #else
        memset(y_512, 0, (VEC_N_512_SIZE_64 >> 3) * sizeof(__m512i));
    #endif

    // Retrieve x, y, pk from secret key
    hqc_secret_key_from_string((__m256i *) x_512, (__m256i *) y_512, pk, sk);

    // Compute v - u.y
    vect_resize(tmp1, PARAM_N, v, PARAM_N1N2);
    vect_mul(tmp3_512, y_512, u_512);
    vect_add(tmp2, tmp1, (uint64_t *) tmp3_512, VEC_N_256_SIZE_64);

    #ifdef VERBOSE
        printf("\n\nu: "); vect_print((uint64_t *) u_512, VEC_N_SIZE_BYTES);
        printf("\n\nv: "); vect_print(v, VEC_N1N2_SIZE_BYTES);
        printf("\n\ny: "); vect_print((uint64_t *) y_512, VEC_N_SIZE_BYTES);
        printf("\n\nv - u.y: "); vect_print(tmp2, VEC_N_SIZE_BYTES);
    #endif

    // Compute m by decoding v - u.y
    code_decode(m, tmp2);
}
