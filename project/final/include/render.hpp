#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "constants.h"
#include "group.hpp"
#include "hit.hpp"
#include "hit_kdtree.hpp"
#include "image.hpp"
#include "light.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "utils.hpp"
using namespace std;
static Vector3f ptColor(Ray ray, const Scene& scene) {
    Group* group = scene.getGroup();
    int depth = 0;
    Vector3f color(0, 0, 0), cf(1, 1, 1);
    while (true) {
        if (++depth > TRACE_DEPTH || cf.max() < 1e-3) return color;
        // 判断camRay是否和场景有交点,返回最近交点的数据,存储在hit中.
        Hit hit;
        if (!group->intersect(ray, hit)) {
            color += scene.getBackgroundColor();
            return color;
        }

        // Path Tracing
        ray.origin += ray.direction * hit.t;
        Material* material = hit.material;
        Vector3f refColor(hit.color), N(hit.normal);

        // Emission
        color += material->emission * cf;
        cf = cf * refColor;
        float type = RND2;
        if (type <= material->type.x()) {  // diffuse
            ray.direction = diffDir(N);
        } else if (type <=
                   material->type.x() + material->type.y()) {  // specular
            float cost = Vector3f::dot(ray.direction, N);
            ray.direction = (ray.direction - N * (cost * 2)).normalized();
        } else {  // refraction
            float n = material->refr;
            float R0 = ((1.0 - n) * (1.0 - n)) / ((1.0 + n) * (1.0 + n));
            if (Vector3f::dot(N, ray.direction) > 0) {  // inside the medium
                N.negate();
                n = 1 / n;
            }
            n = 1 / n;
            float cost1 = -Vector3f::dot(N, ray.direction);  // cosine theta_1
            float cost2 =
                1.0 - n * n * (1.0 - cost1 * cost1);  // cosine theta_2
            float Rprob = R0 + (1.0 - R0) * pow(1.0 - cost1,
                                                5.0);  // Schlick-approximation
            if (cost2 > 0 && RND2 > Rprob) {           // refraction direction
                ray.direction =
                    ((ray.direction * n) + (N * (n * cost1 - sqrt(cost2))))
                        .normalized();
            } else {  // reflection direction
                ray.direction = (ray.direction + N * (cost1 * 2));
            }
        }
    }
}

static Vector3f rcColor(Ray ray, const Scene& scene) {
    Group* group = scene.getGroup();
    int depth = 0;
    Vector3f color(0, 0, 0);
    // 判断camRay是否和场景有交点,返回最近交点的数据,存储在hit中.
    Hit hit;
    if (!group->intersect(ray, hit)) {
        color += scene.getBackgroundColor();
        return color;
    }
    for (int li = 0; li < scene.getNumLights(); ++li) {
        Light* light = scene.getLight(li);
        Vector3f L, lightColor;
        // 获得光照强度
        light->getIllumination(ray.pointAtParameter(hit.getT()), L, lightColor);
        // 计算局部光强
        color += hit.getMaterial()->phongShade(ray, hit, L, lightColor);
    }
}

class PathTracer {
   public:
    const Scene& scene;
    int samps;
    const char* fout;
    Vector3f (*radiance)(Ray ray, const Scene& scene);
    PathTracer(const Scene& scene, int samps, const char* method,
               const char* fout)
        : scene(scene), samps(samps), fout(fout) {
        if (!strcmp(method, "rc"))
            radiance = rcColor;
        else if (!strcmp(method, "pt"))
            radiance = ptColor;
        else {
            cout << "Unknown method: " << method << endl;
            exit(1);
        }
    }

    void render() {
        Camera* camera = scene.getCamera();
        int w = camera->getWidth(), h = camera->getHeight();
        cout << "Width: " << w << " Height: " << h << endl;
        Image outImg(w, h);
        time_t start = time(NULL);
#pragma omp parallel for schedule(dynamic, 1)  // OpenMP
        for (int y = 0; y < h; ++y) {
            float elapsed = (time(NULL) - start), progress = (1. + y) / h;
            fprintf(stderr, "\rRendering (%d spp) %5.2f%% Time: %.2f/%.2f sec",
                    samps, progress * 100., elapsed, elapsed / progress);
            for (int x = 0; x < w; ++x) {
                Vector3f color = Vector3f::ZERO;
                for (int s = 0; s < samps; ++s) {
                    Ray camRay =
                        camera->generateRay(Vector2f(x + RND, y + RND));
                    color += radiance(camRay, scene);
                }
                outImg.SetPixel(x, y, color / samps);
            }
        }
        outImg.SaveBMP(fout);
    }
};

class SPPM {
   public:
    const Scene& scene;
    int numRounds, numPhotons, ckpt_interval;
    std::string outdir;
    int w, h;
    Camera* camera;
    vector<Hit*> hitPoints;
    HitKDTree* hitKDTree;
    vector<Object3D*> illuminants;
    Group* group;
    SPPM(const Scene& scene, int numRounds, int numPhotons, int ckpt,
         const char* dir)
        : scene(scene),
          numRounds(numRounds),
          numPhotons(numPhotons),
          ckpt_interval(ckpt),
          outdir(dir) {
        camera = scene.getCamera();
        group = scene.getGroup();
        illuminants = group->getIlluminant();
        w = camera->getWidth();
        h = camera->getHeight();
        hitKDTree = nullptr;
        for (int u = 0; u < w; ++u)
            for (int v = 0; v < h; ++v) hitPoints.push_back(new Hit());
        cout << "Width: " << w << " Height: " << h << endl;
    }

    ~SPPM() {
        for (int u = 0; u < w; ++u)
            for (int v = 0; v < h; ++v) delete hitPoints[u * w + v];
        delete hitKDTree;
    }

    void forward(Ray ray, Hit* hit) {
        int depth = 0;
        Vector3f attenuation(1, 1, 1);
        while (true) {
            if (++depth > TRACE_DEPTH || attenuation.max() < 1e-3) return;
            hit->t = INF;
            if (!group->intersect(ray, *hit)) {
                hit->fluxLight += hit->attenuation*scene.getBackgroundColor();
                return;
            }
            ray.origin += ray.direction * (*hit).t;
            Material* material = (*hit).material;
            Vector3f N(hit->normal);
            float type = RND2;
            if (type <= material->type.x()) {  // Diffuse
                hit->attenuation = attenuation * hit->color;
                hit->fluxLight += hit->attenuation * material->emission;
                return;
            } else if (type <= material->type.x() + material->type.y()) {
                float cost = Vector3f::dot(ray.direction, N);
                ray.direction = (ray.direction - N * (cost * 2)).normalized();
            } else {
                float n = material->refr;
                float R0 = ((1.0 - n) * (1.0 - n)) / ((1.0 + n) * (1.0 + n));
                if (Vector3f::dot(N, ray.direction) > 0) {  // inside the medium
                    N.negate();
                    n = 1 / n;
                }
                n = 1 / n;
                float cost1 =
                    -Vector3f::dot(N, ray.direction);  // cosine theta_1
                float cost2 =
                    1.0 - n * n * (1.0 - cost1 * cost1);  // cosine theta_2
                float Rprob =
                    R0 + (1.0 - R0) * pow(1.0 - cost1,
                                          5.0);   // Schlick-approximation
                if (cost2 > 0 && RND2 > Rprob) {  // refraction direction
                    ray.direction =
                        ((ray.direction * n) + (N * (n * cost1 - sqrt(cost2))))
                            .normalized();
                } else {  // reflection direction
                    ray.direction =
                        (ray.direction + N * (cost1 * 2)).normalized();
                }
            }
            attenuation = attenuation * hit->color;
        }
    }

    void backward(Ray ray, const Vector3f& color, long long int seed=-1) {
        int depth = 0;
        Vector3f attenuation = color * Vector3f(250, 250, 250);
        while (true) {
            if (++depth > TRACE_DEPTH || attenuation.max() < 1e-3) return;
            Hit hit;
            if (!group->intersect(ray, hit)) return;
            ray.origin += ray.direction * hit.t;
            Material* material = hit.material;
            Vector3f N(hit.normal);
            float type = RND2;
            if (type <= material->type.x()) {  // Diffuse
                hitKDTree->update(hitKDTree->root, hit.p, attenuation,
                                  ray.direction);
                ray.direction = diffDir(N, -1, seed);
            } else if (type <= material->type.x() + material->type.y()) {
                float cost = Vector3f::dot(ray.direction, N);
                ray.direction = (ray.direction - N * (cost * 2)).normalized();
            } else {
                float n = material->refr;
                float R0 = ((1.0 - n) * (1.0 - n)) / ((1.0 + n) * (1.0 + n));
                if (Vector3f::dot(N, ray.direction) > 0) {  // inside the medium
                    N.negate();
                    n = 1 / n;
                }
                n = 1 / n;
                float cost1 =
                    -Vector3f::dot(N, ray.direction);  // cosine theta_1
                float cost2 =
                    1.0 - n * n * (1.0 - cost1 * cost1);  // cosine theta_2
                float Rprob =
                    R0 + (1.0 - R0) * pow(1.0 - cost1,
                                          5.0);   // Schlick-approximation
                if (cost2 > 0 && RND2 > Rprob) {  // refraction direction
                    ray.direction =
                        ((ray.direction * n) + (N * (n * cost1 - sqrt(cost2))))
                            .normalized();
                } else {  // reflection direction
                    ray.direction =
                        (ray.direction + N * (cost1 * 2)).normalized();
                }
            }
            attenuation = attenuation * hit.color;
        }
    }

    void render() {
        time_t start = time(NULL);
        Vector3f color = Vector3f::ZERO;
        for (int round = 0; round < numRounds; ++round) {
            float elapsed = (time(NULL) - start),
                  progress = (1. + round) / numRounds;
            fprintf(stderr,
                    "\rRendering (%d/%d Rounds) %5.2f%% Time: %.2f/%.2f sec\n",
                    round + 1, numRounds, progress * 100., elapsed,
                    elapsed / progress);
#pragma omp parallel for schedule(dynamic, 1)
            for (int x = 0; x < w; ++x) {
                for (int y = 0; y < h; ++y) {
                    Ray camRay =
                        camera->generateRay(Vector2f(x + RND, y + RND));
                    hitPoints[x * h + y]->reset(-camRay.direction);
                    forward(camRay, hitPoints[x * h + y]);
                }
            }
            setHitKDTree();
            int photonsPerLight = numPhotons / illuminants.size();
// photon tracing pass
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < photonsPerLight; ++i) {
                for (int j = 0;j < illuminants.size(); ++j) {
                    // cout << i << " "<< j << " In" <<endl;
                    Ray ray = illuminants[j]->randomRay(-1, (long long)round * numPhotons + (round + 1) * w * h + i);
                    // cout << i << " "<< j << " Out" <<endl;
                    backward(ray, illuminants[j]->material->emission, (long long)round * numPhotons + i);
                } 
            }
            if ((round + 1) % ckpt_interval == 0) {
                char filename[100];
                sprintf(filename, "ckpt-%d.bmp", round + 1);
                save(filename, round + 1, numPhotons);
            }
        }
        save("result.bmp", numRounds, numPhotons);
    }

    void save(std::string filename, int numRounds, int numPhotons) {
        Image outImg(w, h);
        for (int u = 0; u < w; ++u)
            for (int v = 0; v < h; ++v) {
                Hit* hit = hitPoints[u * h + v];
                outImg.SetPixel(
                    u, v,
                    hit->flux / (M_PI * hit->r2 * numPhotons * numRounds) +
                        hit->fluxLight / numRounds);
            }
        outImg.SaveBMP((outdir + "/" + filename).c_str());
    }

    void setHitKDTree() {
        if (hitKDTree) delete hitKDTree;
        hitKDTree = new HitKDTree(&hitPoints);
    }
};

#endif  // !PATH_TRACER_H