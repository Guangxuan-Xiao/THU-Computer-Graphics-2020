#ifndef HIT_KDTREE_H
#define HIT_KDTREE_H
#include <algorithm>

#include "constants.h"
#include "hit.hpp"

class HitKDTreeNode {
   public:
    Hit *hit;
    Vector3f min, max;
    float maxr2;
    HitKDTreeNode *ls, *rs;
};

inline float sqr(float a) { return a * a; }

class HitKDTree {
    int n;
    Hit **hits;
    HitKDTreeNode *build(int l, int r, int d) {
        HitKDTreeNode *p = new HitKDTreeNode;
        p->min = Vector3f(INF, INF, INF);
        p->max = -p->min;
        p->maxr2 = 0;
        for (int i = l; i <= r; ++i) {
            p->min = minE(p->min, hits[i]->p);
            p->max = maxE(p->max, hits[i]->p);
            p->maxr2 = std::max(p->maxr2, hits[i]->r2);
        }

        int m = l + r >> 1;
        if (d == 0)
            std::nth_element(hits + l, hits + m, hits + r + 1, cmpHitX);
        else if (d == 1)
            std::nth_element(hits + l, hits + m, hits + r + 1, cmpHitY);
        else
            std::nth_element(hits + l, hits + m, hits + r + 1, cmpHitZ);
        p->hit = hits[m];
        if (l <= m - 1)
            p->ls = build(l, m - 1, (d + 1) % 3);
        else
            p->ls = nullptr;
        if (m + 1 <= r)
            p->rs = build(m + 1, r, (d + 1) % 3);
        else
            p->rs = nullptr;
        return p;
    }

    void del(HitKDTreeNode *p) {
        if (p->ls) del(p->ls);
        if (p->rs) del(p->rs);
        delete p;
    }

   public:
    HitKDTreeNode *root;
    HitKDTree(vector<Hit *> *hits) {
        n = hits->size();
        this->hits = new Hit *[n];
        for (int i = 0; i < n; ++i) this->hits[i] = (*hits)[i];
        root = build(0, n - 1, 0);
    }
    ~HitKDTree() {
        if (!root) return;
        del(root);
        delete[] hits;
    }

    void update(HitKDTreeNode *p, const Vector3f &photon,
                const Vector3f &attenuation, const Vector3f &d) {
        if (!p) return;
        float mind = 0, maxd = 0;
        if (photon.x() > p->max.x()) mind += sqr(photon.x() - p->max.x());
        if (photon.x() < p->min.x()) mind += sqr(p->min.x() - photon.x());
        if (photon.y() > p->max.y()) mind += sqr(photon.y() - p->max.y());
        if (photon.y() < p->min.y()) mind += sqr(p->min.y() - photon.y());
        if (photon.z() > p->max.z()) mind += sqr(photon.z() - p->max.z());
        if (photon.z() < p->min.z()) mind += sqr(p->min.z() - photon.z());
        if (mind > p->maxr2) return;
        if ((photon - p->hit->p).squaredLength() <= p->hit->r2) {
            Hit *hp = p->hit;
            float factor = (hp->n * ALPHA + ALPHA) / (hp->n * ALPHA + 1.);
            Vector3f dr = d - hp->normal * (2 * Vector3f::dot(d, hp->normal));
            hp->n++;
            hp->r2 *= factor;
            hp->flux = (hp->flux + hp->attenuation * attenuation) * factor;
        }
        if (p->ls) update(p->ls, photon, attenuation, d);
        if (p->rs) update(p->rs, photon, attenuation, d);
        p->maxr2 = p->hit->r2;
        if (p->ls && p->ls->hit->r2 > p->maxr2) p->maxr2 = p->ls->hit->r2;
        if (p->rs && p->rs->hit->r2 > p->maxr2) p->maxr2 = p->rs->hit->r2;
    }

    static bool cmpHitX(Hit *a, Hit *b) { return a->p.x() < b->p.x(); }

    static bool cmpHitY(Hit *a, Hit *b) { return a->p.y() < b->p.y(); }

    static bool cmpHitZ(Hit *a, Hit *b) { return a->p.z() < b->p.z(); }
};
#endif