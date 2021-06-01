The folder Karat3KaratRec contains the following files:

* **gf2x.c** : AVX2 source code of the 3-way split Karatsuba algorithm. Final multiplication is done using 
the **_mm_clmulepi64_si128**  instruction.
* **gf2x_AVX512.c** : AVX512 source code of the 3-way split Karatsuba algorithm. Final multiplication is done using the **_mm512_clmulepi64_epi128** instruction.
* **Karat3.c** : main program to measure the performances of the above multiplications.

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

First configure the msr-tools and disable the turbo-boost feature. As root, run:
```console
bash measure.sh
```

Next as a normal user, run :

```console
make bench
```

to get the performances of all the above multiplications for degrees : 1535, 3071, 6143, 12287, 24575, 49151, 98301.

Or run :

```console
make TEST=1 SIZE=size bench
./Karat3
```
where *size* is one of the size above mentioned to get the performances for the selected size.
 
