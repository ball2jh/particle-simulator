#!/bin/bash

test_duration=7

echo "Running visual tests..."

for n in {1000..2500..500}
do
    timeout "${test_duration}s" ./app_serial -n "${n}" -s 0.005 &
    timeout "${test_duration}s" ./app -n "${n}" -s 0.005 &
    timeout "${test_duration}s" ./app -n "${n}" -s 0.005 -e &
    wait
done

for n in {5000..15000..5000}
do
    timeout "${test_duration}s" ./app_serial -n "${n}" -s 0.002 &
    timeout "${test_duration}s" ./app -n "${n}" -s 0.002 &
    timeout "${test_duration}s" ./app -n "${n}" -s 0.002 -e &
    wait
done

echo "Tests completed."