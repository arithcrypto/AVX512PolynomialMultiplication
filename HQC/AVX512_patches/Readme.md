This repository contains :

* **hqc-optimized-implementation_2021-06-06.zip** : official HQC update for the round 3 of the NIST standardization process 2021/06/06
* **measure_official_release** : Makefile and C-source code to measure the performances of the HQC official release (Optimized_Implementation)
* **patches_AVX512** : Patches fot the official release oh HQC to apply new multiplication process for hqc-128, hqc-192, hqc-256 to obtain an AVX512 version of the HQC package.

**How to measure the performances of the official release ?**

First unzip the HQC release:
```console
unzip hqc-optimized-implementation_2021-06-06.zip
```
**IMPORTANT** : Our work and the measurements only concern the **Optimized_Implementation folder**.

Copy the **Makefile** and **main_bench.c** from the folder **measure_official_release** into the appropriate folders :
```console
cp Makefile Optimized_Implementation/hqc-128
cp Makefile Optimized_Implementation/hqc-192
cp Makefile Optimized_Implementation/hqc-256

cp main_bench.c Optimized_Implementation/hqc-128/src/
cp main_bench.c Optimized_Implementation/hqc-192/src/
cp main_bench.c Optimized_Implementation/hqc-256/src/
```
Do not forget to run the script **measure.sh** located at the top folder of this repository
```console
sudo bash measure.sh
```

To execute a bench :
```console
cd Optimized_Implementation/hqc-size/
make bench
bin/bench
```
where *size* is 128, 192 or 256.

**How to apply our patches ?**

see README of **patches_AVX512** folder.
