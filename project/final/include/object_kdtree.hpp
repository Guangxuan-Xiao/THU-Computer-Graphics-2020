#ifndef OBJECTKDTREE_H
#define OBJECTKDTREE_H
#include <vecmath.h>

#include <map>
#include <vector>

#include "bound.hpp"
#include "hit.hpp"
#include "object3d.hpp"
using std::map;
using std::vector;
class ObjectKDTreeNode {
   public:
    Vector3f min, max;
    vector<Object3D*>* faces;
    ObjectKDTreeNode *ls, *rs;
    int l, r;
    bool inside(Object3D* face) {
        Vector3f faceMin = face->min();
        Vector3f faceMax = face->max();
        return (faceMin.x() < max.x() ||
                faceMin.x() == max.x() && faceMin.x() == faceMax.x()) &&
               (faceMax.x() > min.x() ||
                faceMax.x() == min.x() && faceMin.x() == faceMax.x()) &&
               (faceMin.y() < max.y() ||
                faceMin.y() == max.y() && faceMin.y() == faceMax.y()) &&
               (faceMax.y() > min.y() ||
                faceMax.y() == min.y() && faceMin.y() == faceMax.y()) &&
               (faceMin.z() < max.z() ||
                faceMin.z() == max.z() && faceMin.z() == faceMax.z()) &&
               (faceMax.z() > min.z() ||
                faceMax.z() == min.z() && faceMin.z() == faceMax.z());
    }
};

class ObjectKDTree {
    int n;
    Vector3f** vertices;
    ObjectKDTreeNode* build(int depth, int d, vector<Object3D*>* faces,
                            const Vector3f& min, const Vector3f& max) {
        ObjectKDTreeNode* p = new ObjectKDTreeNode;
        p->min = min;
        p->max = max;
        Vector3f maxL, minR;
        if (d == 0) {
            maxL =
                Vector3f((p->min.x() + p->max.x()) / 2, p->max.y(), p->max.z());
            minR =
                Vector3f((p->min.x() + p->max.x()) / 2, p->min.y(), p->min.z());
        } else if (d == 1) {
            maxL =
                Vector3f(p->max.x(), (p->min.y() + p->max.y()) / 2, p->max.z());
            minR =
                Vector3f(p->min.x(), (p->min.y() + p->max.y()) / 2, p->min.z());
        } else {
            maxL =
                Vector3f(p->max.x(), p->max.y(), (p->min.z() + p->max.z()) / 2);
            minR =
                Vector3f(p->min.x(), p->min.y(), (p->min.z() + p->max.z()) / 2);
        }
        p->faces = new vector<Object3D*>;
        for (auto face : *faces)
            if (p->inside(face)) p->faces->push_back(face);

        const int max_faces = 128;
        const int max_depth = 24;

        if (p->faces->size() > max_faces && depth < max_depth) {
            p->ls = build(depth + 1, (d + 1) % 3, p->faces, min, maxL);
            p->rs = build(depth + 1, (d + 1) % 3, p->faces, minR, max);

            vector<Object3D*>*faceL = p->ls->faces, *faceR = p->rs->faces;
            map<Object3D*, int> cnt;
            for (auto face : *faceL) cnt[face]++;
            for (auto face : *faceR) cnt[face]++;
            p->ls->faces = new vector<Object3D*>;
            p->rs->faces = new vector<Object3D*>;
            p->faces->clear();
            for (auto face : *faceL)
                if (cnt[face] == 1)
                    p->ls->faces->push_back(face);
                else
                    p->faces->push_back(face);
            for (auto face : *faceR)
                if (cnt[face] == 1) p->rs->faces->push_back(face);
        } else
            p->ls = p->rs = nullptr;
        return p;
    }

    void getFaces(ObjectKDTreeNode* p, vector<Object3D*>* faces) {
        p->l = faces->size();
        for (auto face : *(p->faces)) faces->push_back(face);
        p->r = faces->size();
        if (p->ls) getFaces(p->ls, faces);
        if (p->rs) getFaces(p->rs, faces);
    }

   public:
    ObjectKDTreeNode* root;
    vector<Object3D*>* faces;
    ObjectKDTree(vector<Object3D*>* faces) {
        Vector3f min = Vector3f(INF, INF, INF);
        Vector3f max = -min;
        for (auto face : *faces) {
            min = minE(min, face->min());
            max = maxE(max, face->max());
        }
        root = build(1, 0, faces, min, max);
        this->faces = new vector<Object3D*>;
        getFaces(root, this->faces);
    }

    float cuboidIntersect(ObjectKDTreeNode* p, const Ray& ray) const {
        float t = INF;
        if (!p) return t;
        AABB(p->min, p->max).intersect(ray, t);
        return t;
    }

    bool intersect(const Ray& ray, Hit& hit) const {
        Object3D* nextFace = nullptr;
        return intersect(root, ray, nextFace, hit);
    }

    bool intersect(ObjectKDTreeNode* p, const Ray& ray, Object3D*& nextFace,
                   Hit& hit) const {
        bool flag = false;
        for (int i = 0; i < p->faces->size(); ++i)
            if ((*p->faces)[i]->intersect(ray, hit)) {
                nextFace = (*p->faces)[i];
                flag = true;
            }
        float tl = cuboidIntersect(p->ls, ray),
              tr = cuboidIntersect(p->rs, ray);
        if (tl < tr) {
            if (hit.t <= tl) return flag;
            if (p->ls) flag |= intersect(p->ls, ray, nextFace, hit);
            if (hit.t <= tr) return flag;
            if (p->rs) flag |= intersect(p->rs, ray, nextFace, hit);
        } else {
            if (hit.t <= tr) return flag;
            if (p->rs) flag |= intersect(p->rs, ray, nextFace, hit);
            if (hit.t <= tl) return flag;
            if (p->ls) flag |= intersect(p->ls, ray, nextFace, hit);
        }
        return flag;
    }
};

#endif  // !OBJECTKDTREE_H