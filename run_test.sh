#!/bin/bash

echo "Running tests..."
# timeout 5s ./app_serial -n 10000 -s 0.001
timeout 5s ./app -n 10000 -s 0.001