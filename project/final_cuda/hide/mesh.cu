#include "mesh.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

__device__ bool Mesh::intersect(const Ray &r, float t_min, float t_max,
                                Hit &rec) const {
    float tb;
    if (!aabb.intersect(r, tb)) return false;
    if (tb > t_max) return false;
    Hit temp_rec;
    bool flag = false;
    float closest_so_far = t_max;
    for (int i = 0; i < (int)t.size(); i++) {
        TriangleIndex &triIndex = t[triId];
        Triangle triangle(v[triIndex[0]], v[triIndex[1]], v[triIndex[2]],
                          material);
        triangle.normal = n[triId];
        result |= triangle.intersect(r, h, tmin);
        if (list[i]->intersect(r, t_min, closest_so_far, temp_rec)) {
            flag = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
}

__device__ Mesh::Mesh(const char *filename, Material *material)
    : Object(material) {
    thrust::device_vector<Vec3> v;
    thrust::device_vector<TriangleIndex> t;
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    char bslash = '/', space = ' ';
    std::string tok;
    int texID;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vec3 vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            v.push_back(vec);
            aabb.updateBound(vec);
        } else if (tok == fTok) {
            if (line.find(bslash) != std::string::npos) {
                std::replace(line.begin(), line.end(), bslash, space);
                std::stringstream facess(line);
                TriangleIndex trig;
                facess >> tok;
                for (int ii = 0; ii < 3; ii++) {
                    facess >> trig[ii] >> texID;
                    trig[ii]--;
                }
                t.push_back(trig);
            } else {
                TriangleIndex trig;
                for (int ii = 0; ii < 3; ii++) {
                    ss >> trig[ii];
                    trig[ii]--;
                }
                t.push_back(trig);
            }
        } else if (tok == texTok) {
            float texcoord[2];
            ss >> texcoord[0];
            ss >> texcoord[1];
        }
    }
    num_triangles = t.size();
    triangles = new Triangle *[num_triangles];
    for (int triId = 0; triId < num_triangles; ++triId) {
        TriangleIndex triIndex = t[triId];
        triangles[triId] =
            new Triangle(v[triIndex[0]], v[triIndex[1]], v[triIndex[2]]);
    }

    f.close();
}
