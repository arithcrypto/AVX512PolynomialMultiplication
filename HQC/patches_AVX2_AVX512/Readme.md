This repository contains :

* **AVX2** : patches to apply to the AVX2 Optimized Implementation of the official HQC update for the round 3 of the NIST standardization process 2020/10/01. It concerns only hqc-128 and hqc-192.
* **AVX512** : patches to apply to the AVX2 Optimized Implementation of the official HQC update for the round 3 of the NIST standardization process 2020/10/01, to get an AVX512 version of HQC multiplication process.

**How to apply patches ?**

First unzip the HQC release:
```console
unzip hqc-submission_2020-10-01.zip
```
**IMPORTANT** : Our work and the measurements only concern the **Optimized_Implementation folder**.

Copy the **Makefile**  from the appropriate patch folder into one of the HQC subfolder :
```console
cp AVXversion/Makefile Optimized_Implementation/hqc-size
```
where *version* is 2 or 512 and *size* is 128, 192 (for AVX2 and AVX512) or 256 (for AVX512).

Then copy the source codes from one of the patch subfolder into the corresponding **src** subfolder of HQC.

```console
cp AVXversion/hqc-size/* Optimized_Implementation/hqc-size/src/
```
where *version* is 2 or 512 and *size* is 128, 192 (for AVX2 and AVX512) or 256 (for AVX512).

Do not forget to run the script **measure.sh** located at the top folder of this repository
```bash
sudo bash measure.sh
```

To execute a bench :
```console
cd Optimized_Implementation/hqc-size/
make bench
bin/bench
```
where *version* is 2 or 512 and *size* is 128, 192 (for AVX2 and AVX512) or 256 (for AVX512).


