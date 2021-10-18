## Description
This folder contains the source code of the multiplication procedure described in *Chen, M.-S., Chou, T. and Krausz, M. 2021. Optimizing BIKE for the Intel Haswell and ARM Cortex-M4. IACR Transactions on Cryptographic Hardware and Embedded Systems. 2021, 3 (Jul. 2021), 97–124. DOI:https://doi.org/10.46586/tches.v2021.i3.97-124.*

#### rkara3_mul_avx2.c
Source code of the multiplication written by Ming-Shin Chen and Tung Chou modified to include timings measures. This source code is the one provided in the Supercop package (20210604 release).
Other C files are mandatories in order to compile this code.

**How to run ?**

First disable the turbo-boost feature. In a shell, run:
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
The output lines are  (for size = 12288, 24576, 12352 and 24704)

*rkara3_mul_size* : original multiplication provided by Chen and Chou

*karat_mult3_size* : AVX2 source code multiplication provided in HQC source code

*rkara3_mul_size_bis* : Modification of Chen Chou multiplication using *karat_mult_3*  
