#ifndef CURVE_HPP
#define CURVE_HPP

#include <vecmath.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "object3d.hpp"

// TODO (PA3): Implement Bernstein class to compute spline basis function.
//       You may refer to the python-script for implementation.

// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in
// all the data.
using namespace std;
struct CurvePoint {
    Vector3f V;  // Vertex
    Vector3f T;  // Tangent  (unit)
};

class Curve : public Object3D {
   protected:
    std::vector<Vector3f> controls;

   public:
    explicit Curve(std::vector<Vector3f> points)
        : controls(std::move(points)) {}

    bool intersect(const Ray &r, Hit &h, float tmin) override { return false; }

    std::vector<Vector3f> &getControls() { return controls; }

    void discretize(int resolution, std::vector<CurvePoint> &data) {
        resolution *= n / k;
        data.resize(resolution);
        for (int i = 0; i < resolution; ++i) {
            data[i].T = Vector3f::ZERO;
            data[i].V = Vector3f::ZERO;
            double mu =
                ((double)i / resolution) * (range[1] - range[0]) + range[0];
            int bpos = upper_bound(t.begin(), t.end(), mu) - t.begin() - 1;
            vector<double> s(k + 2, 0), ds(k + 1, 1);
            s[k] = 1;
            for (int p = 1; p <= k; ++p) {
                for (int ii = k - p; ii < k + 1; ++ii) {
                    int i = ii + bpos - k;
                    double w1, dw1, w2, dw2;
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
                data[i].V += controls[-lsk + j] * s[j];
                data[i].T += controls[-lsk + j] * ds[j];
            }
        }
    }

    void drawGL() override {
        Object3D::drawGL();
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 0);
        glBegin(GL_LINE_STRIP);
        for (auto &control : controls) {
            glVertex3fv(control);
        }
        glEnd();
        glPointSize(4);
        glBegin(GL_POINTS);
        for (auto &control : controls) {
            glVertex3fv(control);
        }
        glEnd();
        std::vector<CurvePoint> sampledPoints;
        discretize(30, sampledPoints);
        glColor3f(1, 1, 1);
        glBegin(GL_LINE_STRIP);
        for (auto &cp : sampledPoints) {
            glVertex3fv(cp.V);
        }
        glEnd();
        glPopAttrib();
    }
    void pad() {
        int tSize = t.size();
        tpad.resize(tSize + k);
        for (int i = 0; i < tSize; ++i) tpad[i] = t[i];
        for (int i = 0; i < k; ++i) tpad[i + tSize] = t.back();
    }
    int n, k;
    std::vector<double> t;
    std::vector<double> tpad;
    double range[2];
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
        for (int i = 0; i < n + k + 1; ++i) t[i] = (double)i / (n + k);
        pad();
        range[0] = t[k];
        range[1] = t[n];
    }
};

#endif  // CURVE_HPP
