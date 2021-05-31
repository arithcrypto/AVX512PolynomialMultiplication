This repository contains AVX512 implementations using the **VPCLMULQDQ** instruction for the 
* schoolbook multiplication in <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle"> of polynomials of degree up to 131071.

The folder KaratRec contains the following files:
* SB256.c : source code of the schoolbook multiplication of polynomials of degree up to 255 using AVX512 instruction set. For degree > 255, Recursive Karatsuba is used.
* Karat256.c : source code of the Karatsuba multiplication of polynomials of degree up to 255 using AVX512 instruction set. For degree > 255, Recursive Karatsuba is used.
* DGK.c : source code of the multiplication of polynomials of degree up to 255 described in *"N. Drucker, S. Gueron, V. Krasnov, Fast Multiplication of binary polynomials with the forthcoming vectorized vpclmulqdq instruction, ARITH'25, 2018"*. 
* AVX2.c : source code of the schoolbook multiplication of polynomials of degree up to 127 using AVX2 instruction set. For degree > 255, Recursive Karatsuba is used.
* KaratRec.c : main program to measure the performances of the above multiplications.

**Prerequisites**

To run the tests you must have :
* an AVX512 processor with the VPCLMULQDQ instruction. From a shell, just run :
```console
grep vpclmulqdq /proc/cpuinfo
``` 
to see if this feature is available.
* the gf2x library (version >= 1.2) installed on your system.

**How to run ?**


