The folder Toom3KaratRec contains the following files:

* **gf2xmul_AVX2.c** : AVX2 source code of recursive steps of the classical Karatsuba algorithm. Final multiplication is done using 
the **_mm_clmulepi64_si128**  instruction.
* **gf2xmul_AVX512.c** : AVX512 source code of recursive steps of the classical Karatsuba algorithm. Final multiplication is done using the **_mm512_clmulepi64_epi128** instruction.
* **ToomCookMult_AVX2.c** : AVX2 source code of the Toom-Cook multiplication algorithm.
* **ToomCookMult_AVX512.c** : AVX512 source code of the Toom-Cook multiplication algorithm.

* **ToomCook.c** : main program to measure the performances of the above multiplications.

**Prerequisites**

To run the tests you must have :
* an AVX512 processor with the VPCLMULQDQ instruction. From a shell, just run :
```console
grep vpclmulqdq /proc/cpuinfo
``` 
to see if this feature is available
* the gf2x library (version >= 1.2) 
* gcc 10.2.0
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

to get the performances of all the above multiplications for degrees : 24191, 48767, 97919.

