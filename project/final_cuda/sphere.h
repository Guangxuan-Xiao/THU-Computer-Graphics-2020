#ifndef SPHEREH
#define SPHEREH

#include <cfloat>
#include <cmath>

#include "object.h"
class Sphere : public Object {
   public:
    __device__ Sphere() {}
    __device__ Sphere(Vec3 cen, float r, Material* m, Bump* bump = nullptr)
        : Object(m), center(cen), radius(r), bump(bump){};
    __device__ virtual bool intersect(const Ray& r, float tmin, float tmax,
                                      Hit& rec) const {
        Vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < tmax && temp > tmin) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                Vec3 normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                getUV(normal, rec.u, rec.v);
                rec.normal = getNormal(normal, rec.p, rec.u, rec.v);
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < tmax && temp > tmin) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                Vec3 normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                getUV(normal, rec.u, rec.v);
                rec.normal = getNormal(normal, rec.p, rec.u, rec.v);
                return true;
            }
        }
        return false;
    }

    __device__ void getUV(const Vec3& n, float& u, float& v) const {
        u = 0.5 + atan2(n.x(), n.z()) / (2 * M_PI);
        v = 0.5 - asin(n.y()) / M_PI;
    }

    __device__ Vec3 getNormal(const Vec3& n, const Vec3& p, float u,
                              float v) const {
        if (!bump) return n;
        float grad[2] = {0};
        float f = bump->getDisturb(u, v, grad);
        if (fabs(f) < FLT_EPSILON) return n;
        float phi = u * 2 * M_PI, theta = M_PI - v * M_PI;
        Vec3 pu(-p.z(), 0, p.x()),
            pv(p.y() * cos(phi), -radius * sin(theta), p.y() * sin(phi));
        if (pu.squared_length() < FLT_EPSILON) return n;
        return cross(pu + n * grad[0] / (2 * M_PI), pv + n * grad[1] / M_PI)
            .normalized();
    }

    Vec3 center;
    float radius;
    Bump* bump;
};

#endif
