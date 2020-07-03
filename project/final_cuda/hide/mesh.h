#ifndef MESH_H
#define MESH_H

#include <thrust/device_vector.h>

#include "bound.h"
#include "object.h"
#include "triangle.h"
#include "vec3.h"
class Mesh : public Object {
   public:
    __device__ Mesh(const char *filename, Material *m);
    __device__ ~Mesh() { delete[] triangles; }
    struct TriangleIndex {
        __device__ TriangleIndex() {
            x[0] = 0;
            x[1] = 0;
            x[2] = 0;
        }
        __device__ int &operator[](const int i) { return x[i]; }
        int x[3];
    };

    Triangle **triangles;
    int num_triangles;
    __device__ bool intersect(const Ray &r, float t_min, float t_max,
                              Hit &rec) const override;

   private:
    AABB aabb;
};

#endif
