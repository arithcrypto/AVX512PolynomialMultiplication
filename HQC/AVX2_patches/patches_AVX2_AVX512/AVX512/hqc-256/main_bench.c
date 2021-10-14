#define _GNU_SOURCE

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/syscall.h>

#include "rng.h"
#include "api.h"
#include "parameters.h"

#include <stdint.h>

#define NB_TEST 1000
#define NB_SAMPLES 100



inline static uint64_t
cpucyclesStart (void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("CPUID\n\t"
			"RDTSC\n\t"
			"mov %%edx, %0\n\t"
			"mov %%eax, %1\n\t":"=r" (hi), "=r" (lo)::"%rax",
			"%rbx", "%rcx", "%rdx");

  return ((uint64_t) lo) ^ (((uint64_t) hi) << 32);
}



inline static uint64_t
cpucyclesStop (void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("RDTSCP\n\t"
			"mov %%edx, %0\n\t"
			"mov %%eax, %1\n\t"
			"CPUID\n\t":"=r" (hi), "=r" (lo)::"%rax", "%rbx",
			"%rcx", "%rdx");

  return ((uint64_t) lo) ^ (((uint64_t) hi) << 32);
}



int
main ()
{

  unsigned char pk[PUBLIC_KEY_BYTES];
  unsigned char sk[SECRET_KEY_BYTES];
  unsigned char ct[CIPHERTEXT_BYTES];
  unsigned char ss1[SHARED_SECRET_BYTES];
  unsigned char ss2[SHARED_SECRET_BYTES];

  unsigned char seed[48];
  syscall (SYS_getrandom, seed, 48, 0);
  randombytes_init (seed, NULL, 256);


  unsigned long long timer, t1, t2;
  unsigned long long keygen_mean = 0, encaps_mean = 0, decaps_mean = 0;
  int failures = 0;

  // Cache memory heating
  printf ("\nkem_keypair : Heating Cache ....\n");
  for (size_t i = 0; i < NB_TEST; i++)
    {
      crypto_kem_keypair (pk, sk);
    }

  printf ("kem_keypair : Measure");
  // Measurement
  for (size_t i = 0; i < NB_SAMPLES; i++)
    {
      crypto_kem_keypair (pk, sk);
      timer = 0;

      for (size_t j = 0; j < NB_TEST; j++)
	{
	  t1 = cpucyclesStart ();
	  crypto_kem_keypair (pk, sk);
	  t2 = cpucyclesStop ();

	  timer += t2 - t1;
	}
      if ((i % 10) == 0)
	{
	  printf (".");
	  fflush (stdout);
	}
      keygen_mean += timer / NB_TEST;
    }



  // Cache memory heating
  printf ("\nkem_enc : Heating Cache ....\n");
  for (size_t i = 0; i < NB_TEST; i++)
    {
      crypto_kem_enc (ct, ss1, pk);
    }

  // Measurement
  printf ("kem_enc : Measure");
  for (size_t i = 0; i < NB_SAMPLES; i++)
    {
      crypto_kem_keypair (pk, sk);
      timer = 0;

      for (size_t j = 0; j < NB_TEST; j++)
	{
	  t1 = cpucyclesStart ();
	  crypto_kem_enc (ct, ss1, pk);
	  t2 = cpucyclesStop ();

	  timer += t2 - t1;
	}
      if ((i % 10) == 0)
	{
	  printf (".");
	  fflush (stdout);
	}
      encaps_mean += timer / NB_TEST;
    }


  printf ("\nkem_dec : Heating Cache ...\n");
  // Cache memory heating
  for (size_t i = 0; i < NB_TEST; i++)
    {
      crypto_kem_dec (ss2, ct, sk);
    }


  // Measurement
  printf ("kem_dec : Measure");
  for (size_t i = 0; i < NB_SAMPLES; i++)
    {
      crypto_kem_keypair (pk, sk);
      crypto_kem_enc (ct, ss1, pk);
      if (crypto_kem_dec (ss2, ct, sk))
	failures++;
      timer = 0;

      if (memcmp (ss1, ss2, SHARED_SECRET_BYTES))
	failures++;

      for (size_t j = 0; j < NB_TEST; j++)
	{
	  t1 = cpucyclesStart ();
	  crypto_kem_dec (ss2, ct, sk);
	  t2 = cpucyclesStop ();

	  timer += t2 - t1;
	}
      if ((i % 10) == 0)
	{
	  printf (".");
	  fflush (stdout);
	}
      decaps_mean += timer / NB_TEST;
    }


  printf ("\n\n");
  printf ("\nhqc");
  printf ("\n  Failures: %i", failures);
  printf ("\n  Keygen: %lld CPU cycles", keygen_mean / NB_SAMPLES);
  printf ("\n  Encaps: %lld CPU cycles", encaps_mean / NB_SAMPLES);
  printf ("\n  Decaps: %lld CPU cycles", decaps_mean / NB_SAMPLES);

  printf ("\n\n");
  return 0;
}
