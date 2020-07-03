#include <curand_kernel.h>
#include <float.h>
#include <time.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "camera.h"
#include "material.h"
#include "object_list.h"
#include "render.h"
#include "sphere.h"
#include "texture.h"
#include "vec3.h"
const int tx = 16;
const int ty = 16;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        // cudaDeviceReset();
        // exit(99);
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

#define RND (curand_uniform(&local_rand_state))

int main(int argc, char *argv[]) {
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cerr << "Argument " << argNum << " is: " << argv[argNum]
                  << std::endl;
    }
    if (argc < 4) {
        std::cerr << "Usage: ./final <input scene file> <output ppm file> <spp>"
                  << std::endl;
        return 1;
    }

    int ns = atoi(argv[3]);
    Scene **scene;
    checkCudaErrors(cudaMallocManaged((void **)&scene, sizeof(Scene *)));
    *scene = new Scene(argv[1]);
    checkCudaErrors(cudaDeviceSynchronize());

    curandState *d_rand_state0;
    checkCudaErrors(
        cudaMalloc((void **)&d_rand_state0, 1 * sizeof(curandState)));
    checkCudaErrors(
        cudaMalloc((void **)&d_rand_state0, 1 * sizeof(curandState)));
    rand_init<<<1, 1>>>(d_rand_state0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int nx = (*scene)->width;
    int ny = (*scene)->height;
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns
              << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Vec3);

    // allocate FB
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(
        cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "ObjList: " << *(*scene)->objList
              << " Camera: " << *(*scene)->camera << std::endl;
    render<<<blocks, threads>>>(fb, nx, ny, ns, (*scene)->objList,
                                (*scene)->camera, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Elapsed " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::ofstream ppm;
    ppm.open(argv[2]);
    ppm << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            ppm << ir << " " << ig << " " << ib << "\n";
        }
    }
    ppm.close();

    // clean up
    // checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(scene));
    checkCudaErrors(cudaFree(d_rand_state0));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
}
