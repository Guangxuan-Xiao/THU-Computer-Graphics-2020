#ifndef OBJECTH
#define OBJECTH

#include "material.h"
#include "ray.h"

class Object {
   public:
    __device__ Object(Material* mat = nullptr) : mat_ptr(mat) {}
    __device__ virtual bool intersect(const Ray& r, float t_min, float t_max,
                                      Hit& rec) const = 0;
    Material* mat_ptr;
};

#endif
