#!/bin/bash

test_duration=5

echo "Running frame rate tests..."

total_fps=0

# Serial Test 1000-2500
echo "Serial Tests 1000-2500 particles, 0.004 size"
for n in {1000..25000..2400}
do
    avg_fps=$(stdbuf -oL timeout "${test_duration}s" ./app_serial -n "1000" -s 0.002 | awk '{sum+=$1} END {print sum/NR}')
    echo "Serial frame rate $n particles 0.004 size = $avg_fps"
done

# CUDA Test 1000-2500
echo "CUDA Tests 1000-2500 particles, 0.004 size"
for n in {1000..25000..2400}
do
    avg_fps=$(stdbuf -oL timeout "${test_duration}s" ./app -n "$n" -s 0.002 | awk '{sum+=$1} END {print sum/NR}')
    echo "CUDA frame rate $n particles 0.004 size = $avg_fps"
done


echo "Tests completed."
