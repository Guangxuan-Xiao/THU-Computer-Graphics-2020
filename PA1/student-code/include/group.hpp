#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>

// 存储物体列表：球、平面，组织成一个列表
// TODO: Implement Group - add data structure to store a list of Object*
class Group : public Object3D {

public:

    Group() {

    }

    explicit Group (int num_objects) {

    }

    ~Group() override {

    }

    // 对列表里所有物体都求一遍交点
    bool intersect(const Ray &r, Hit &h, float tmin) override {

    }

    void addObject(int index, Object3D *obj) {

    }

    int getGroupSize() {

    }

private:

};

#endif
	
