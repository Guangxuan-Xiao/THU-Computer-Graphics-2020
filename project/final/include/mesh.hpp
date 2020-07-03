#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "Vector2f.h"
#include "Vector3f.h"
#include "bound.hpp"
#include "object3d.hpp"
#include "object_kdtree.hpp"
#include "ray.hpp"
#include "triangle.hpp"
#include "utils.hpp"
class Mesh : public Object3D {
   public:
    Mesh(const char *filename, Material *m) : Object3D(m) {
        std::vector<TriangleIndex> vIdx, tIdx, nIdx;
        std::vector<Vector3f> v, vn;
        std::vector<Vector2f> vt;
        std::ifstream f;
        f.open(filename);
        if (!f.is_open()) {
            std::cout << "Cannot open " << filename << "\n";
            return;
        }
        std::string line;
        std::string vTok("v");
        std::string fTok("f");
        std::string vnTok("vn");
        std::string texTok("vt");
        std::string bslash("/"), space(" ");
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
                Vector3f vec;
                ss >> vec[0] >> vec[1] >> vec[2];
                v.push_back(vec);
                aabb.updateBound(vec);
            } else if (tok == fTok) {
                bool tFlag = 1, nFlag = 1;
                TriangleIndex vId, tId, nId;
                for (int i = 0; i < 3; ++i) {
                    std::string str;
                    ss >> str;
                    std::vector<std::string> id = split(str, bslash);
                    vId[i] = atoi(id[0].c_str()) - 1;
                    if (id.size() > 1) {
                        tId[i] = atoi(id[1].c_str()) - 1;
                    }
                    if (id.size() > 2) {
                        nId[i] = atoi(id[2].c_str()) - 1;
                    }
                }
                vIdx.push_back(vId);
                tIdx.push_back(tId);
                nIdx.push_back(nId);
                // if (line.find(bslash) != std::string::npos) {
                //     std::replace(line.begin(), line.end(), '/', ' ');
                //     std::stringstream facess(line);
                //     TriangleIndex trig;
                //     facess >> tok;
                //     for (int ii = 0; ii < 3; ii++) {
                //         facess >> trig[ii] >> texID;
                //         trig[ii]--;
                //     }
                //     vIdx.push_back(trig);
                // } else {
                //     TriangleIndex trig;
                //     for (int ii = 0; ii < 3; ii++) {
                //         ss >> trig[ii];
                //         trig[ii]--;
                //     }
                //     vIdx.push_back(trig);
                // }
            } else if (tok == texTok) {
                Vector2f texcoord;
                ss >> texcoord[0];
                ss >> texcoord[1];
                vt.push_back(texcoord);
            } else if (tok == vnTok) {
                Vector3f vec;
                ss >> vec[0] >> vec[1] >> vec[2];
                vn.push_back(vec);
            }
        }
        f.close();
        for (int triId = 0; triId < (int)vIdx.size(); ++triId) {
            TriangleIndex &vIndex = vIdx[triId];
            triangles.push_back((Object3D *)new Triangle(
                v[vIndex[0]], v[vIndex[1]], v[vIndex[2]], m));
            // if (tIdx.size()) {
            TriangleIndex &tIndex = tIdx[triId];
            if (tIndex.valid())
                ((Triangle *)triangles.back())
                    ->setVT(vt[tIndex[0]], vt[tIndex[1]], vt[tIndex[2]]);
            // }
            // if (nIdx.size()) {
            TriangleIndex &nIndex = nIdx[triId];
            if (nIndex.valid())
                ((Triangle *)triangles.back())
                    ->setVNorm(vn[nIndex[0]], vn[nIndex[1]], vn[nIndex[2]]);
            // }
        }
        kdTree = new ObjectKDTree(&triangles);
    }

    ~Mesh() {
        for (int i = 0; i < triangles.size(); ++i) delete triangles[i];
        delete kdTree;
    }

    std::vector<std::string> split(std::string str, std::string pattern) {
        std::string::size_type pos;
        std::vector<std::string> result;
        str += pattern;
        int size = str.size();

        for (int i = 0; i < size; i++) {
            pos = str.find(pattern, i);
            if (pos < size) {
                std::string s = str.substr(i, pos - i);
                result.push_back(s);
                i = pos + pattern.size() - 1;
            }
        }
        return result;
    }

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = -1;
            x[1] = -1;
            x[2] = -1;
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front
        // face
        int x[3]{};
        bool valid() { return x[0] != -1 && x[1] != -1 && x[2] != -1; }
    };

    vector<Object3D *> triangles;
    bool intersect(const Ray &r, Hit &h) override {
        float tb;
        if (!aabb.intersect(r, tb)) return false;
        if (tb > h.getT()) return false;
        // kd-tree search
        bool flag = kdTree->intersect(r, h);
        return flag;
        // sequential search
        // return sequentialSearch(r, h);
    }

    bool sequentialSearch(const Ray &r, Hit &h) {
        bool result = false;
        for (auto triangle : triangles) result |= triangle->intersect(r, h);
        return result;
    }

    Vector3f min() const override { return aabb.bounds[0]; }
    Vector3f max() const override { return aabb.bounds[1]; }
    Vector3f center() const override {
        return (aabb.bounds[0] + aabb.bounds[1]) / 2;
    }
    vector<Object3D *> getFaces() override { return {(Object3D *)this}; }
    Ray randomRay(int axis=-1, long long int seed=0) const override {
        int trig = random(axis, seed) * triangles.size();
        return triangles[trig]->randomRay(axis, seed);
    }

   private:
    AABB aabb;
    ObjectKDTree *kdTree;
};

#endif
