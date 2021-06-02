#!/bin/bash
# Load the msr-tools module in the kernel
modprobe msr
echo "-1" > /proc/sys/kernel/perf_event_paranoid
# Allow all users to use the drpmc
echo 2 | dd of=/sys/devices/cpu/rdpmc
echo 2 | dd of=/sys/bus/event_source/devices/cpu/rdpmc
# In order to use the three fixed function counters
wrmsr -a 0x38d 0x0333
# Disable turbo boost
echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo

