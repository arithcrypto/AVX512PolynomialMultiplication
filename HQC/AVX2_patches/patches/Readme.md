This repository contains patches to apply to the AVX2 Optimized Implementation of the official HQC update for the round 3 of the NIST standardization process 2020/10/01. It concerns only hqc-128 and hqc-192.

**How to apply patches ?**

First unzip the HQC release:
```console
unzip hqc-submission_2020-10-01.zip
```
**IMPORTANT** : Our work and the measurements only concern the **Optimized_Implementation folder**.

Copy the **Makefile**  from the appropriate patch folder into one of the HQC subfolder :
```console
cp Makefile Optimized_Implementation/hqc-size
```
where *size* is 128 or 192.

Then copy the source codes from one of the patch subfolder into the corresponding **src** subfolder of HQC.

```console
cp hqc-size/* Optimized_Implementation/hqc-size/src/
```
where *size* is 128 or 192.

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
where *size* is 128 or 192.


