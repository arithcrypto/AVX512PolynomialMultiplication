This repository contains AVX512 implementations using the **VPCLMULQDQ** instruction for the :
* schoolbook multiplication of polynomials over <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle">
* Karatsuba multiplication of polynomials over <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle">
* Toom-Cook multiplication of polynomials over <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle">

As an illustration of the performances obtained with this instruction, the HQC folder contains source files to patch the HQC optimized implementation (2020/10/01 release) which has been submitted to the third round of the NIST Post-Quantum Cryptography process initiated to standardize one or more quantum-resistant public-key cryptographic algorithms. On our reference platform (i7-1165G7 @ 2.80GHz), this leads to speed-ups up to 15.5%.
