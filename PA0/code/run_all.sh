#!/usr/bin/env bash

# If project not ready, generate cmake file.
if [[ ! -d build ]]; then
    mkdir -p build
    cd build
    cmake ..
    cd ..
fi

# Build project.
cd build
make -j
cd ..

# Run all testcases.
# You can comment some lines to disable the run of specific examples.
mkdir -p output
bin/PA0 testcases/canvas01_basic.txt output/canvas01.bmp
bin/PA0 testcases/canvas02_emoji.txt output/canvas02.bmp
