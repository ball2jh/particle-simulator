#!/bin/bash

test_duration=5

echo "Running frame rate tests..."

total_fps=0
num_tests=0

# Serial Test 1000-2500
for n in {1000..2500..500}
do
    output=$(stdbuf -oL timeout "${test_duration}s" ./app_serial -n "$n" -s 0.004 | awk '{sum+=$1} END {print sum/NR}')

    total_fps=$(bc <<< "$total_fps + $output")

    # output the frame rate
    avg_fps=$(bc <<< "scale=2; $total_fps / $test_duration")
    echo "Serial frame rate $n particles 0.004 size = $output"
done

# CUDA Test 1000-2500
for n in {1000..2500..500}
do
    output=$(stdbuf -oL timeout "${test_duration}s" ./app -n "$n" -s 0.004 | awk '{sum+=$1} END {print sum/NR}')

    total_fps=$(bc <<< "$total_fps + $output")

    # output the frame rate
    avg_fps=$(bc <<< "scale=2; $total_fps / $test_duration")
    echo "CUDA frame rate $n particles 0.004 size = $output"
done


echo "Tests completed."
