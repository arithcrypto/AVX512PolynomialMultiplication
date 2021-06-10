This repository contains :

* **hqc-submission_2020-10-01.zip** : official HQC update for the round 3 of the NIST standardization process 2020/10/01
* **measure_official_release** : Makefile and C-source code to measure the performance of the HQC official release (Optimized_Implementation)
* **patches_AVX2_AVX512** : Patches to apply to the official release to apply new multiplication process for hqc-128 and hqc-192 (AVX2), and patches to obtain an AVX_512 version of the HQC package.

**How to run ?**

First unzip the HQC release:
```console
unzip hqc-submission_2020-10-01.zip
```
**IMPORTANT** : Our work and the measurements done only concerns the **Optimzed_Implementation folder**.

Next copy the **Makefile** and **main_bench.c** from the folder **measure_official_release** into the appropriate folders :
```console
cp Makefile Optimized_Implementation/hqc-128
cp Makefile Optimized_Implementation/hqc-192
cp Makefile Optimized_Implementation/hqc-256

cp main_bench.c Optimized_Implementation/hqc-128/src/
cp main_bench.c Optimized_Implementation/hqc-192/src/
cp main_bench.c Optimized_Implementation/hqc-256/src/
```

To execute a bench :
```console
cd Optimized_Implementation/hqc-*size*/
make bench
bin/bench
```
where *size* is 128, 192 or 256.




