#ifndef GROUP_H
#define GROUP_H

#include <iostream>
#include <vector>

#include "hit.hpp"
#include "object3d.hpp"
#include "ray.hpp"

// 存储物体列表：球、平面，组织成一个列表
// TODO: Implement Group - add data structure to store a list of Object*
class Group : public Object3D {
   public:
    Group() {}

    explicit Group(int num_objects) : objList(num_objects) {}

    ~Group() override {}

    // 对列表里所有物体都求一遍交点
    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool flag = false;
        for (auto obj : objList)
            if (obj) flag |= obj->intersect(r, h, tmin);
        return flag;
    }

    void addObject(int index, Object3D *obj) {
        objList.insert(objList.begin() + index, obj);
    }

    int getGroupSize() { return objList.size(); }

   private:
    std::vector<Object3D *> objList;
};

#endif
