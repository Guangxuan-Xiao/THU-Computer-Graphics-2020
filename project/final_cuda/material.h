#ifndef MATERIALH
#define MATERIALH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "hit.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

#define RANDVEC3                                                             \
    Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), \
         curand_uniform(local_rand_state))

class Material {
   public:
    __device__ Material(const Vec3& a, const Vec3& t, const Vec3& e, float refr,
                        Texture* texture = nullptr)
        : albedo(a), type(t), emission(e), refr(refr), texture(texture) {
        type = type / (type.x() + type.y() + type.z());
    }

    __device__ void scatter(const Ray& r_in, const Hit& rec, Vec3& attenuation,
                            Ray& scattered,
                            curandState* local_rand_state) const {
        float t = curand_uniform(local_rand_state);
        attenuation = getAlbedo(rec.u, rec.v);
        if (t <= type.x()) {  // Diffuse
            Vec3 target =
                rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            scattered = Ray(rec.p, (target - rec.p));
        } else if (t <= type.x() + type.y()) {  // Mirror
            Vec3 reflected = reflect(normalized(r_in.direction()), rec.normal);
            scattered = Ray(rec.p, reflected);
        } else {  // Refractor
            Vec3 outward_normal;
            Vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            Vec3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = refr;
                cosine = dot(r_in.direction(), rec.normal) /
                         r_in.direction().length();
                cosine = sqrt(1.0f - refr * refr * (1 - cosine * cosine));
            } else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / refr;
                cosine = -dot(r_in.direction(), rec.normal) /
                         r_in.direction().length();
            }
            if (refract(r_in.direction(), outward_normal, ni_over_nt,
                        refracted))
                reflect_prob = schlick(cosine, refr);
            else
                reflect_prob = 1.0f;
            if (curand_uniform(local_rand_state) < reflect_prob)
                scattered = Ray(rec.p, reflected);
            else
                scattered = Ray(rec.p, refracted);
        }
    }

    __device__ inline Vec3 getAlbedo(float u, float v) const {
        if (!texture)
            return albedo;
        else
            return texture->getColor(u, v);
    }
    __device__ Vec3 random_in_unit_sphere(curandState* local_rand_state) const {
        Vec3 p;
        do {
            p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
        } while (p.squared_length() >= 1.0f);
        return p;
    }

    __device__ Vec3 reflect(const Vec3& v, const Vec3& n) const {
        return v - 2.0f * dot(v, n) * n;
    }

    __device__ float schlick(float cosine, float refr) const {
        float r0 = (1.0f - refr) / (1.0f + refr);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }

    __device__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt,
                            Vec3& refracted) const {
        Vec3 uv = normalized(v);
        float dt = dot(uv, n);
        float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
        if (discriminant > 0) {
            refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
            return true;
        } else
            return false;
    }

    Vec3 albedo, type, emission;
    Texture* texture;
    float refr;
};

// class Diffusive : public Material {
//    public:
//     __device__ Diffusive(const Vec3& a) : albedo(a) {}
//     __device__ virtual bool scatter(const Ray& r_in, const Hit& rec,
//                                     Vec3& attenuation, Ray& scattered,
//                                     curandState* local_rand_state) const {
//         Vec3 target =
//             rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
//         scattered = Ray(rec.p, target - rec.p);
//         attenuation = albedo;
//         return true;
//     }

//     Vec3 albedo;
// };

// class Mirror : public Material {
//    public:
//     __device__ Mirror(const Vec3& a, float f) : albedo(a) {
//         if (f < 1)
//             fuzz = f;
//         else
//             fuzz = 1;
//     }
//     __device__ virtual bool scatter(const Ray& r_in, const Hit& rec,
//                                     Vec3& attenuation, Ray& scattered,
//                                     curandState* local_rand_state) const {
//         Vec3 reflected = reflect(normalized(r_in.direction()), rec.normal);
//         scattered = Ray(
//             rec.p, reflected + fuzz *
//             random_in_unit_sphere(local_rand_state));
//         attenuation = albedo;
//         return (dot(scattered.direction(), rec.normal) > 0.0f);
//     }
//     Vec3 albedo;
//     float fuzz;
// };

// class Refractor : public Material {
//    public:
//     __device__ Refractor(float ri) : refr(ri) {}
//     __device__ virtual bool scatter(const Ray& r_in, const Hit& rec,
//                                     Vec3& attenuation, Ray& scattered,
//                                     curandState* local_rand_state) const {
//         Vec3 outward_normal;
//         Vec3 reflected = reflect(r_in.direction(), rec.normal);
//         float ni_over_nt;
//         attenuation = Vec3(1.0, 1.0, 1.0);
//         Vec3 refracted;
//         float reflect_prob;
//         float cosine;
//         if (dot(r_in.direction(), rec.normal) > 0.0f) {
//             outward_normal = -rec.normal;
//             ni_over_nt = refr;
//             cosine =
//                 dot(r_in.direction(), rec.normal) /
//                 r_in.direction().length();
//             cosine = sqrt(1.0f - refr * refr * (1 - cosine * cosine));
//         } else {
//             outward_normal = rec.normal;
//             ni_over_nt = 1.0f / refr;
//             cosine =
//                 -dot(r_in.direction(), rec.normal) /
//                 r_in.direction().length();
//         }
//         if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
//             reflect_prob = schlick(cosine, refr);
//         else
//             reflect_prob = 1.0f;
//         if (curand_uniform(local_rand_state) < reflect_prob)
//             scattered = Ray(rec.p, reflected);
//         else
//             scattered = Ray(rec.p, refracted);
//         return true;
//     }

//     float refr;
// };
#endif
