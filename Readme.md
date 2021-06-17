This repository contains AVX512 implementations using the **VPCLMULQDQ** instruction and AVX2 implementations using the **PCLMULQDQ** instruction for the multiplication of polynomials over <img src="https://render.githubusercontent.com/render/math?math=\mathbb F_{2}[X]/(X^n-1)" valign="middle">. 

As an illustration of the performances obtained with the **VPCLMULQDQ** instruction, the HQC folder contains source files to patch the HQC AVX2 optimized implementation (2020/10/01 release, cf. https://pqc-hqc.org/) which has been submitted to the third round of the [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography "NIST Post-Quantum Cryptography") process initiated to standardize one or more quantum-resistant public-key cryptographic algorithms. On our reference platform (i7-1165G7 @ 2.80GHz), this leads to speed-ups up to 12%.

**Important** : 
Read the *Readme.md* of each subfolder to obtain more details on how to compile the source codes. The script **measure.sh** has to be run before any test. It allows to count the instructions of a running program and it disables the Turbo boost feature so as to obtain stable measures.

**Test platform** : All the codes have been tested on a Dell Inspiron 7506 2n1, i7-1165G7 @ 2.80GHz, gcc 10.2.0 and gf2x v1.3 library.


