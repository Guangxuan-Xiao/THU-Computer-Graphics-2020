#ifndef VEC3H
#define VEC3H

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include <cfloat>
#include <iostream>
#define VEC3ZERO (Vec3(0.f, 0.f, 0.f))
#define VEC3UP (Vec3(0.f, 1.f, 0.f))
#define VEC3RIGHT (Vec3(1.f, 0.f, 0.f))
#define VEC3FORWARD (Vec3(0.f, 0.f, -1.f))
class Vec3 {
   public:
    __host__ __device__ Vec3() { e[0] = e[1] = e[2] = 0; }
    __host__ __device__ Vec3(float e0, float e1, float e2) {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ Vec3(const Vec3 &v) {
        e[0] = v[0];
        e[1] = v[1];
        e[2] = v[2];
    }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vec3 &operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const {
        return Vec3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float &operator[](int i) { return e[i]; };

    __host__ __device__ inline bool operator==(const Vec3 &v2) const {
        return fabs(e[0] - v2[0]) < FLT_EPSILON &&
               fabs(e[1] - v2[1]) < FLT_EPSILON &&
               fabs(e[2] - v2[2]) < FLT_EPSILON;
    }
    __host__ __device__ inline Vec3 &operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3 &operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3 &operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3 &operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3 &operator*=(const float t);
    __host__ __device__ inline Vec3 &operator/=(const float t);

    __host__ __device__ inline float length() const {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }
    __host__ __device__ inline float squared_length() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    __host__ __device__ inline void normalize();
    __host__ __device__ inline Vec3 normalized() const {
        float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        return Vec3(e[0] * k, e[1] * k, e[2] * k);
    }

    float e[3];
};

inline std::istream &operator>>(std::istream &is, Vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const Vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void Vec3::normalize() {
    float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3 operator/(float t, const Vec3 &v) {
    return Vec3(t / v.e[0], t / v.e[1], t / v.e[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline Vec3 &Vec3::operator+=(const Vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator*=(const Vec3 &v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator/=(const Vec3 &v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator-=(const Vec3 &v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator/=(const float t) {
    float k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline Vec3 normalized(Vec3 v) { return v / v.length(); }

#endif
