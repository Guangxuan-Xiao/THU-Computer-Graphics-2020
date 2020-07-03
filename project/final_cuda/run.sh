#!/usr/bin/env bash
SAMPLES=100
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
time bin/PA1 testcases/sample.txt output/sample.ppm $SAMPLES
python ppmconverter.py --input output/sample.ppm --output output/sample.png
