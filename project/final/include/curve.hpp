#ifndef CURVE_HPP
#define CURVE_HPP

#include <vecmath.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "constants.h"
#include "object3d.hpp"
#include "constants.h"
using namespace std;
struct CurvePoint {
    Vector3f V;  // Vertex
    Vector3f T;  // Tangent  (unit)
};

class Curve {
   protected:
    std::vector<Vector3f> controls;

   public:
    explicit Curve(std::vector<Vector3f> points) : controls(std::move(points)) {
        ymin = INF;
        ymax = -INF;
        radius = 0;
        for (auto pt : controls) {
            ymin = min(pt.y(), ymin);
            ymax = max(pt.y(), ymax);
            radius = max(radius, fabs(pt.x()));
            radius = max(radius, fabs(pt.z()));
        }
    }

    std::vector<Vector3f> &getControls() { return controls; }

    CurvePoint getPoint(float mu) {
        CurvePoint pt;
        int bpos = upper_bound(t.begin(), t.end(), mu) - t.begin() - 1;
        vector<float> s(k + 2, 0), ds(k + 1, 1);
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
        s.pop_back();
        int lsk = k - bpos, rsk = bpos + 1 - n;
        if (lsk > 0) {
            for (int i = lsk; i < s.size(); ++i) {
                s[i - lsk] = s[i];
                ds[i - lsk] = ds[i];
            }
            s.resize(s.size() - lsk);
            ds.resize(ds.size() - lsk);
            lsk = 0;
        }
        if (rsk > 0) {
            if (rsk < s.size()) {
                s.resize(s.size() - rsk);
                ds.resize(ds.size() - rsk);
            } else {
                s.clear();
                ds.clear();
            }
        }
        for (int j = 0; j < s.size(); ++j) {
            pt.V += controls[-lsk + j] * s[j];
            pt.T += controls[-lsk + j] * ds[j];
        }
        return pt;
    }

    void discretize(int resolution, std::vector<CurvePoint> &data) {
        resolution *= n / k;
        data.resize(resolution);
        for (int i = 0; i < resolution; ++i) {
            float mu =
                ((float)i / resolution) * (range[1] - range[0]) + range[0];
            data[i] = getPoint(mu);
        }
    }

    void pad() {
        int tSize = t.size();
        tpad.resize(tSize + k);
        for (int i = 0; i < tSize; ++i) tpad[i] = t[i];
        for (int i = 0; i < k; ++i) tpad[i + tSize] = t.back();
    }

    int n, k;
    std::vector<float> t;
    std::vector<float> tpad;
    float ymin, ymax, radius;
    float range[2];
};

class BezierCurve : public Curve {
   public:
    explicit BezierCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4 || points.size() % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
        n = controls.size();
        k = n - 1;
        range[0] = 0;
        range[1] = 1;
        t.resize(2 * n);
        for (int i = 0; i < n; ++i) {
            t[i] = 0;
            t[i + n] = 1;
        }
        pad();
    }
};

class BsplineCurve : public Curve {
   public:
    BsplineCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4) {
            printf(
                "Number of control points of BspineCurve must be more than "
                "4!\n");
            exit(0);
        }
        n = controls.size();
        k = 3;
        t.resize(n + k + 1);
        for (int i = 0; i < n + k + 1; ++i) t[i] = (float)i / (n + k);
        pad();
        range[0] = t[k];
        range[1] = t[n];
    }
};

#endif  // CURVE_HPP
