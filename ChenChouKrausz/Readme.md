## Description


#### rkara3_mul_avx2.c
Source code of the multiplication written by Ming-Shin Chen and Tung Chou modified to include timings measures.


**How to run ?**

First configure the msr-tools and disable the turbo-boost feature. In a shell, run:
```console
sudo bash measure.sh
```

Next as a normal user, run :

```console
make
```
then run
```console
./rkara3_mul
```
The output lines are  for size = 12288, 24576, 12352 and 24704

*rkara3_mul_size* : original multiplication provided by Chen and Chou

*karat_mult3_size* : AVX2 source code multiplication provided in HQC source code

*rkara3_mul_size_bis* : Modification of Chen Chou multiplication using karat_mult_3  
