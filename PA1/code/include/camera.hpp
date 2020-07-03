#ifndef CAMERA_H
#define CAMERA_H

#include <float.h>
#include <vecmath.h>

#include <cmath>

#include "ray.hpp"

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

// TODO: Implement Perspective camera
// You can add new functions or variables whenever needed.
class PerspectiveCamera : public Camera {
   public:
    PerspectiveCamera(const Vector3f &center, const Vector3f &direction,
                      const Vector3f &up, int imgW, int imgH, float angle)
        : Camera(center, direction, up, imgW, imgH) {
        // angle is in radian.
        cx = imgW / 2;
        cy = imgH / 2;
        fx = cx / tan(angle / 2);
        fy = cy / tan(angle / 2);
    }

    Ray generateRay(const Vector2f &point) override {
        Vector3f d_rc = Vector3f((point[0] - cx) / fx, (point[1] - cy) / fy, 1)
                            .normalized();
        Matrix3f R(horizontal, up, direction);
        return Ray(center, R * d_rc);
    }

   protected:
    float fx, fy, cx, cy;
};

#endif  // CAMERA_H
