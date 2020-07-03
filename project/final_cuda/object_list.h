#ifndef OBJECTLISTTH
#define OBJECTLISTTH

#include "object.h"

class ObjectList : public Object {
   public:
    __device__ ObjectList(Object** l, int n) : list(l), list_size(n) {}

    __device__ ~ObjectList() {
        for (int i = 0; i < list_size; ++i) delete list[i];
        delete[] list;
    }

    __device__ virtual bool intersect(const Ray& r, float t_min, float t_max,
                                      Hit& rec) const {
        Hit temp_rec;
        bool flag = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->intersect(r, t_min, closest_so_far, temp_rec)) {
                flag = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return flag;
    }
    Object** list;
    int list_size;
};
#endif
