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
bin/PA3 testcases/scene01_basic.txt output/scene01.bmp
bin/PA3 testcases/scene04_axes.txt output/scene04.bmp
bin/PA3 testcases/scene06_bunny_1k.txt output/scene06.bmp
bin/PA3 testcases/scene08_core.txt output/scene08.bmp
bin/PA3 testcases/scene09_norm.txt output/scene09.bmp
bin/PA3 testcases/scene10_wineglass.txt output/scene10.bmp
