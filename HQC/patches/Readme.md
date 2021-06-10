This repository contains :

* **AVX2** : patches to apply to the AVX2 Optimized Implemetation of the official HQC update for the round 3 of the NIST standardization process 2020/10/01. It concerns only hqc-128 and hqc-192.
* **AVX512** : patches to apply to the AVX2 Optimized Implemetation of the official HQC update for the round 3 of the NIST standardization process 2020/10/01, to get an AVX512 version oh HQC multiplication process.

**How to apply patches ?**

First unzip the HQC release:
```console
unzip hqc-submission_2020-10-01.zip
```
**IMPORTANT** : Our work and the measurements only concerns the **Optimized_Implementation folder**.

Next copy the **Makefile**  from the appropriate patch folder into one of the HQC subfolder :
```console
cp AVXversion/Makefile Optimized_Implementation/hqc-size
```
where *version* is 2 or 512 and *size* is 128, 192 (for AVX2) or 256 (for AVX512).

Then copy the source codes from the one of the patch subfolder into the corresponding *src* subfolder of HQC.

```console
cp AVXversion/hqc-size/* Optimized_Implementation/hqc-size/src/
```

To execute a bench :
```console
cd Optimized_Implementation/hqc-size/
make bench
bin/bench
```
where *size* is 128, 192 (for AVX2) or 256 (for AVX512)


