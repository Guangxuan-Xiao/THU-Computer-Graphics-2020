#ifndef CURVE_H
#define CURVE_H
#include <cstdlib>
#include <iostream>
#include <utility>

#include "object.h"
#include "vec3.h"
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
struct CurvePoint {
    Vec3 V;  // Vertex
    Vec3 T;  // Tangent  (unit)
};

class Curve : public Object {
   public:
    Vec3 *controls;
    int num_controls;

    __device__ explicit Curve(Vec3 *points, int num_controls)
        : Object(nullptr), controls(points), num_controls(num_controls) {
        ymin = 0x3f3f3f3f;
        ymax = -0x3f3f3f3f;
        radius = 0;
        for (int i = 0; i < num_controls; ++i) {
            ymin = min(controls[i].y(), ymin);
            ymax = max(controls[i].y(), ymax);
            radius = max(radius, fabs(controls[i].x()));
        }
    }
    __device__ ~Curve() {
        delete[] t;
        delete[] tpad;
        delete[] controls;
    }

    __device__ bool intersect(const Ray &r, float t_min, float t_max,
                              Hit &rec) const override {
        return false;
    }

    __device__ Vec3 *getControls() const { return controls; }

    __device__ int upper_bound(float *begin, int len, int e) const {
        int l = 0, r = len - 1, mid;
        while (l <= r) {
            mid = r - (r - l) / 2;
            if (begin[mid] <= e)
                l = mid + 1;
            else
                r = mid - 1;
        }
        return l;
    }

    __device__ CurvePoint getPoint(float mu) const {
        CurvePoint pt;
        int bpos = upper_bound(t, 2 * n, mu) - 1;
        float *s = new float[k + 2], *ds = new float[k + 1];
        int slen = k + 2, dslen = k + 1;
        for (int i = 0; i < k + 1; ++i) ds[i] = 1;
        for (int i = 0; i < k + 2; ++i) s[i] = 0;
        s[k] = 1;
        for (int p = 1; p <= k; ++p) {
            for (int ii = k - p; ii < k + 1; ++ii) {
                int i = ii + bpos - k;
                float w1, dw1, w2, dw2;
                if (tpad[i + p] == tpad[i]) {
                    w1 = mu;
                    dw1 = 1;
                } else {
                    w1 = (mu - tpad[i]) / (tpad[i + p] - tpad[i]);
                    dw1 = 1.0 / (tpad[i + p] - tpad[i]);
                }
                if (tpad[i + p + 1] == tpad[i + 1]) {
                    w2 = 1 - mu;
                    dw2 = -1;
                } else {
                    w2 = (tpad[i + p + 1] - mu) /
                         (tpad[i + p + 1] - tpad[i + 1]);
                    dw2 = -1 / (tpad[i + p + 1] - tpad[i + 1]);
                }
                if (p == k) ds[ii] = (dw1 * s[ii] + dw2 * s[ii + 1]) * p;
                s[ii] = w1 * s[ii] + w2 * s[ii + 1];
            }
        }
        --slen;
        int lsk = k - bpos, rsk = bpos + 1 - n;
        if (lsk > 0) {
            for (int i = lsk; i < slen; ++i) {
                s[i - lsk] = s[i];
                ds[i - lsk] = ds[i];
            }
            slen -= lsk;
            dslen -= lsk;
            lsk = 0;
        }
        if (rsk > 0) {
            if (rsk < slen) {
                slen -= rsk;
                dslen -= rsk;
            } else {
                slen = 0;
                dslen = 0;
            }
        }
        for (int j = 0; j < slen; ++j) {
            pt.V += controls[-lsk + j] * s[j];
            pt.T += controls[-lsk + j] * ds[j];
        }
        delete[] s;
        delete[] ds;
        return pt;
    }

    __device__ void pad() {
        int tSize = 2 * n;
        for (int i = 0; i < tSize; ++i) tpad[i] = t[i];
        for (int i = 0; i < k; ++i) tpad[i + tSize] = t[tSize - 1];
    }

    int n, k;
    float *t;
    float *tpad;
    float ymin, ymax, radius;
    float range[2];
};

class BezierCurve : public Curve {
   public:
    __device__ explicit BezierCurve(Vec3 *points, int num_controls)
        : Curve(points, num_controls) {
        if (num_controls < 4 || num_controls % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
        }
        n = num_controls;
        k = n - 1;
        range[0] = 0;
        range[1] = 1;
        t = new float[2 * n];
        tpad = new float[2 * n + k];
        for (int i = 0; i < n; ++i) {
            t[i] = 0;
            t[i + n] = 1;
        }
        pad();
    }
};

#endif  // CURVE_HPP
