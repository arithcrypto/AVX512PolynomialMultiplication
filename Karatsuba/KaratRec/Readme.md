The folder KaratRec contains the following files:

* **SB256.c** : source code of the schoolbook multiplication of polynomials of degree up to 255 using AVX512 instruction set. For degree > 255, recursive Karatsuba is used.
* **SB512.c** : source code of the schoolbook multiplication of polynomials of degree up to 511 using AVX512 instruction set. For degree > 511, recursive Karatsuba is used.
* **Karat256.c** : source code of the Karatsuba multiplication of polynomials of degree up to 255 using AVX512 instruction set. For degree > 255, recursive Karatsuba is used.
* **Karat512.c** : source code of the Karatsuba multiplication of polynomials of degree up to 511 using AVX512 instruction set and the multiplication 128x4 provided in DGK2.c. For degree > 511, recursive Karatsuba is used.
* **Karat512_SB.c** : source code of the Karatsuba multiplication of polynomials of degree up to 511 using AVX512 instruction using the schoolbook multiplication provided in SB512.c for degree < 128. For degree > 511, recursive Karatsuba is used.
* **DGK.c** : source code of the AVX512 multiplication of polynomials of degree up to 255 described in *"N. Drucker, S. Gueron, V. Krasnov, Fast Multiplication of binary polynomials with the forthcoming vectorized vpclmulqdq instruction, ARITH'25, 2018"*. For degree > 255, recursive Karatsuba is used.
* **DGK2.c** : source code of the AVX512 multiplication of polynomials of degree up to 511 described in *"N. Drucker, S. Gueron, D. Kostic, Fast Polynomial Inversion for Post-Quantum QC-MDPC Cryptography, CSCML 2020, 2020"*. For degree > 511, recursive Karatsuba is used.
* **AVX2.c** : source code of the schoolbook multiplication of polynomials of degree up to 127 using AVX2 instruction set. For degree > 255, Recursive Karatsuba is used.
* **KaratRec.c** : main program to measure the performances of the above multiplications.

**Prerequisites**

To run the tests you must have :
* an AVX512 processor with the VPCLMULQDQ instruction. From a shell, just run :
```console
grep vpclmulqdq /proc/cpuinfo
```
to see if this feature is available.
* the gf2x library (version >= 1.2)
* gcc version 10.2.0
* the msr-tools

**How to run ?**

First configure the msr-tools and disable the turbo-boost feature. In a shell, run:
```console
sudo bash measure.sh
```

Next as a normal user, run :

```console
make bench
```

to get the performances of all the above multiplications for degrees : 1023, 2047, 4095, 16383, 32767, 65535 and 131071.

Or run :

```console
make METHOD=1 bench
./KaratRec
```
where METHOD is either **SB256**, **KARAT256**, **DGK**, or **AVX2** to get the performances of the selected method.
