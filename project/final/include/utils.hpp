#ifndef UTILS_H
#define UTILS_H
#include <cmath>
#include <cstdlib>
#include <random>

#include <vecmath.h>
#define PI 3.1415926536
const int prime[] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,127,131, 137, 139, 149, 151, 157, 163, 167, 173,179, 181, 191, 193, 197, 199, 211, 223, 227, 229};
// Helpers for random number generation
static std::mt19937 mersenneTwister;
static std::uniform_real_distribution<float> uniform;
#define RND (2.0 * uniform(mersenneTwister) - 1.0)
#define RND2 (uniform(mersenneTwister))
inline float clamp(float x)
{
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
inline Vector3f gamma(const Vector3f &v)
{
    const float correct = 1 / 2.2;
    return Vector3f(pow(v.x(), correct), pow(v.y(), correct),
                    pow(v.z(), correct));
}

inline void ons(const Vector3f &v1, Vector3f &v2, Vector3f &v3)
{
    if (std::abs(v1.x()) > std::abs(v1.y()))
    {
        // project to the y = 0 plane and construct a normalized orthogonal
        // vector in this plane
        float invLen = 1.f / sqrtf(v1.x() * v1.x() + v1.z() * v1.z());
        v2 = Vector3f(-v1.z() * invLen, 0.0f, v1.x() * invLen);
    }
    else
    {
        // project to the x = 0 plane and construct a normalized orthogonal
        // vector in this plane
        float invLen = 1.0f / sqrtf(v1.y() * v1.y() + v1.z() * v1.z());
        v2 = Vector3f(0.0f, v1.z() * invLen, -v1.y() * invLen);
    }
    v3 = Vector3f::cross(v1, v2);
}

inline Vector3f hemisphere(float u1, float u2)
{
    const float r = sqrt(1.0 - u1 * u1);
    const float phi = 2 * PI * u2;
    return Vector3f(cos(phi) * r, sin(phi) * r, u1);
}

inline Vector3f cosineHemisphere(float u1, float u2)
{
    const float r = sqrt(u1);
    const float theta = 2 * PI * u2;

    const float x = r * cos(theta);
    const float y = r * sin(theta);

    return Vector3f(x, y, sqrt(std::max(0.0f, 1 - u1)));
}

inline float randomQMC(int axis, long long int seed) {
    int base = prime[axis];
    float f = 1, res = 0;
    while (seed > 0) {
        f /= base;
        res += f * (seed % base);
        seed /= base;
    }
    return res;
}

inline float random(int axis=-1, long long int seed=0) {
    if (axis == -1) return RND2;
    return randomQMC(axis, seed);
}

inline Vector3f diffDir(const Vector3f &norm, int depth=0, long long int seed=0)
{
    Vector3f rotX, rotY;
    ons(norm, rotX, rotY);
    return Matrix3f(rotX, rotY, norm) * cosineHemisphere(random(2*depth+1, seed), random(2*depth+2, seed));
}

#endif // !UTILS_H
