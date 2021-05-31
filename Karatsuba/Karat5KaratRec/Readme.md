The folder Karat5KaratRec contains the following files:

* **gf2x.c** : AVX2 source code of the 5-way split Karatsuba algorithm. Final multiplication is done using 
the **_mm_clmulepi64_si128**  instruction.
* **gf2x_AVX512.c** : AVX512 source code of the 5-way split Karatsuba algorithm. Final multiplication is done using the **_mm512_clmulepi64_epi128** instruction.
* **Karat5.c** : main program to measure the performances of the above multiplications.

**Prerequisites**

To run the tests you must have :
* an AVX512 processor with the VPCLMULQDQ instruction. From a shell, just run :
```console
grep vpclmulqdq /proc/cpuinfo
``` 
to see if this feature is available.
* the gf2x library (version >= 1.2) installed on your system.
* gcc 10.2.0.

**How to run ?**

From a shell, run :

```console
make bench
```

to get the performances of all the above multiplications for degrees : 2559, 5119, 10239, 20479, 40959, 81919.

Or run :

```console
make TEST=1 SIZE=size bench
./Karat5
```
where *size* is one of the size above mentioned to get the performances for the selected size.
 
