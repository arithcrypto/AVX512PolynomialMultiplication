This repository contains patches to apply to the AVX2 Optimized Implementation of the official HQC update for the round 3 of the NIST standardization process 2021/06/06.

**How to apply patches ?**

First unzip the HQC release:
```console
unzip hqc-optimized-implementation_2021-06-06.zip
```
**IMPORTANT** : Our work and the measurements only concern the **Optimized_Implementation folder**.

Copy the **Makefile**  from the appropriate patch folder into one of the HQC subfolder :
```console
cp AVX512/Makefile Optimized_Implementation/hqc-size
```
where *size* is 128, 192 or 256.

Then copy the source codes from one of the patch subfolder into the corresponding **src** subfolder of HQC.

```console
cp AVX512/hqc-size/* Optimized_Implementation/hqc-size/src/
```
where size is 128, 192 or 256.

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
where size is 128, 192 or 256.


