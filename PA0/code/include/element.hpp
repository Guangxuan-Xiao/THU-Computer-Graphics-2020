#pragma once
#define FLT_EPSILON 1.19209290e-7F
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>

#include "image.hpp"
using namespace std;

class Element {
   public:
    virtual void draw(Image &img) = 0;
    virtual ~Element() = default;
    bool legal(Image &img, int x, int y) {
        return x >= 0 && x < img.Width() && y >= 0 && y < img.Height();
    }
};

class Line : public Element {
   public:
    int xA, yA;
    int xB, yB;
    Vector3f color;
    void draw(Image &img) override {
        // FTODO: Implement Bresenham Algorithm
        if (abs(xB - xA) > FLT_EPSILON) {
            if (xA > xB) {
                swap(xA, xB);
                swap(yA, yB);
            }
            int x = xA, y = yA, dx = xB - xA, dy = yB - yA;
            int e = -dx;
            for (int i = 0; i <= dx; ++i) {
                if (!legal(img, x, y)) continue;
                img.SetPixel(x, y, color);
                ++x;
                e += 2 * abs(dy);
                while (e >= 0) {
                    e -= 2 * dx;
                    if (dy > 0)
                        ++y;
                    else if (dy < 0)
                        --y;
                }
            }
        } 
        if (abs(yB - yA) > FLT_EPSILON){
            if (yA > yB) {
                swap(xA, xB);
                swap(yA, yB);
            }
            int x = xA, y = yA, dx = xB - xA, dy = yB - yA;
            int e = -dx;
            for (int i = 0; i <= dy; ++i) {
                if (!legal(img, x, y)) {
                    continue;
                }
                img.SetPixel(x, y, color);
                ++y;
                e += 2 * abs(dx);
                while (e >= 0) {
                    e -= 2 * dy;
                    if (dx > 0)
                        ++x;
                    else if (dx < 0)
                        --x;
                }
            }
        }
        printf(
            "Draw a line from (%d, %d) to (%d, %d) using color (%f, %f, "
            "%f)\n",
            xA, yA, xB, yB, color.x(), color.y(), color.z());
    }
};

class Circle : public Element {
   public:
    int cx, cy;
    int radius;
    Vector3f color;
    void draw(Image &img) override {
        // FTODO: Implement Algorithm to draw a Circle
        int x = 0, y = radius;
        float d = 1.25 - radius;
        circlePoints(img, x, y);
        while (x <= y) {
            if (d < 0) {
                d += 2 * x + 3;
            } else {
                d += 2 * (x - y) + 5;
                --y;
            }
            ++x;
            circlePoints(img, x, y);
        }

        printf(
            "Draw a circle with center (%d, %d) and radius %d using color (%f, "
            "%f, %f)\n",
            cx, cy, radius, color.x(), color.y(), color.z());
    }
    void circlePoints(Image &img, int x, int y) {
        int dx[8] = {x, -x, x, -x, y, -y, y, -y};
        int dy[8] = {y, y, -y, -y, x, x, -x, -x};
        for (int i = 0; i < 8; ++i) {
            int x = cx + dx[i], y = cy + dy[i];
            if (x < 0 || x >= img.Width() || y < 0 || y >= img.Height()) {
                continue;
            }
            img.SetPixel(x, y, color);
        }
    }
};

class Fill : public Element {
    struct Seed {
        int x, y;
        Seed(int x, int y) : x(x), y(y) {}
    };

   public:
    int cx, cy;
    Vector3f color;
    void draw(Image &img) override {
        // TODO: Flood fill
        int dx[4] = {0, 0, 1, -1}, dy[4] = {1, -1, 0, 0};
        Vector3f oldColor = img.GetPixel(cx, cy);
        Seed pt(cx, cy);
        queue<Seed> q;
        q.push(pt);
        img.SetPixel(pt.x, pt.y, color);
        if (color == oldColor) return;
        while (!q.empty()) {
            pt = q.front();
            q.pop();
            for (int i = 0; i < 4; ++i) {
                int newx = pt.x + dx[i], newy = pt.y + dy[i];
                if (legal(img, newx, newy) &&
                    img.GetPixel(newx, newy) == oldColor) {
                    img.SetPixel(newx, newy, color);
                    q.push(Seed(newx, newy));
                }
            }
        }
        printf("Flood fill source point = (%d, %d) using color (%f, %f, %f)\n",
               cx, cy, color.x(), color.y(), color.z());
    }
};