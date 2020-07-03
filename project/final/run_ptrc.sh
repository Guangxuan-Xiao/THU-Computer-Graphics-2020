#!/usr/bin/env bash
SAMPLES=10
METHOD=rc
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
# mkdir -p output
# time bin/PA1 testcases/scene01_basic.txt output/scene01.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene02_cube.txt output/scene02.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene03_sphere.txt output/scene03.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene04_axes.txt output/scene04.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene05_bunny_200.txt output/scene05.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene06_bunny_1k.txt output/scene06_bunny_1k.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene06_bunny_1k_vn.txt output/scene06_bunny_1k_vn.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene07_shine.txt output/scene07.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene08_core.txt output/scene08.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene09_norm.txt output/scene09_norm.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene10_wineglass.txt output/scene10_wineglass.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene10_wineglass_720p.txt output/scene10_wineglass_720p.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene12.txt output/scene12.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene13.txt output/scene13.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene14.txt output/scene14.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene12_720p.txt output/scene12_720p.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene11_smallpt.txt output/scene11.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene15_bunny_200.txt output/scene15_bunny_200.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene16_global_diff.txt output/scene16_global_diff.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene16_global.txt output/scene16_global.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene17_bump.txt output/scene17_bump.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene18_dof.txt output/scene18_dof.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene18_dof_0.txt output/scene18_dof_0.bmp $METHOD $SAMPLES
time bin/PA1 testcases/scene19_sibenik.txt output/scene19_sibenik.bmp $METHOD $SAMPLES
# time bin/PA1 testcases/scene21_livingroom.txt output/scene21_livingroom.bmp $METHOD $SAMPLES