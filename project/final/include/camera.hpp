#ifndef CAMERA_H
#define CAMERA_H

#include <float.h>

#include <cmath>

#include "ray.hpp"
#include "utils.hpp"
#include <vecmath.h>
const float INF_FOCAL_LENGTH = 0x3f3f3f3f;
class Camera {
   public:
    Camera(const Vector3f &center, const Vector3f &direction,
           const Vector3f &up, int imgW, int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up);
        this->up = Vector3f::cross(this->horizontal, this->direction);
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    virtual Ray generateRay(const Vector2f &point) = 0;
    virtual ~Camera() = default;

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    void setCenter(const Vector3f &pos) { this->center = pos; }
    Vector3f getCenter() const { return this->center; }

    void setRotation(const Matrix3f &mat) {
        this->horizontal = mat.getCol(0);
        this->up = -mat.getCol(1);
        this->direction = mat.getCol(2);
    }
    Matrix3f getRotation() const {
        return Matrix3f(this->horizontal, -this->up, this->direction);
    }

    virtual void resize(int w, int h) {
        width = w;
        height = h;
    }

   protected:
    // Extrinsic parameters
    Vector3f center;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;
    // Intrinsic parameters
    int width;
    int height;
};

class PerspectiveCamera : public Camera {
   public:
    float getFovy() const { return fovyd; }

    PerspectiveCamera(const Vector3f &center, const Vector3f &direction,
                      const Vector3f &up, int imgW, int imgH, float angle,
                      float f = 20.0f, float aperture = 1.0f)
        : Camera(center, direction, up, imgW, imgH),
          focalLength(f),
          aperture(aperture) {
        // angle is fovy in radian.
        fovyd = angle / 3.1415 * 180.0;
        fx = fy = (float)height / (2 * tanf(angle / 2));
        cx = width / 2.0f;
        cy = height / 2.0f;
    }

    void resize(int w, int h) override {
        fx *= (float)h / height;
        fy = fx;
        Camera::resize(w, h);
        cx = width / 2.0f;
        cy = height / 2.0f;
    }

    Ray generateRay(const Vector2f &point) override {
        float csx = focalLength * (point.x() - cx) / fx;
        float csy = focalLength * (point.y() - cy) / fy;
        float dx = RND * aperture, dy = RND * aperture;
        Vector3f dir(csx - dx, -csy - dy, focalLength);
        Matrix3f R(horizontal, -up, direction);
        dir = (R * dir).normalized();
        Ray ray(center + horizontal * dx - up * dy, dir);
        return ray;
    }

   protected:
    // Perspective intrinsics
    float fx;
    float fy;
    float cx;
    float cy;
    float fovyd;
    float aperture, focalLength;
};
#endif  // CAMERA_H
