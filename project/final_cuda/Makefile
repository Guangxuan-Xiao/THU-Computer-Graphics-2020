SPP = 100
CUDA_PATH     ?= /usr/local/cuda-10.0
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
NVCC_DBG       = -g -G -O3
# NVCC_DBG       = 

NVCCFLAGS      = $(NVCC_DBG) -m64 #  -Xcompiler -fPIC--expt-relaxed-constexpr -rdc=true 
GENCODE_FLAGS  =

SRCS = main.cu scene.cu
INCS = $(wildcard ./*.h)

final: .o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) main.o scene.o -o final 

.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c scene.cu


output/sample.ppm: final
	mkdir -p output
	rm -f output/sample.ppm
	./final testcases/sample.txt output/sample.ppm $(SPP)

output/sample.jpg: output/sample.ppm
	rm -f output/sample.jpg
	python ppmconverter.py --input output/sample.ppm --output output/sample.jpg

output/sample.png: output/sample.ppm
	rm -f output/sample.png
	python ppmconverter.py --input output/sample.ppm --output output/sample.png

profile_basic: final
	nvprof 	./final testcases/sample.txt output/sample.ppm $(SPP)

# use nvprof --query-metrics
profile_metrics: final
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./final testcases/sample.txt output/sample.ppm $(SPP)

clean:
	rm -f *.o final
