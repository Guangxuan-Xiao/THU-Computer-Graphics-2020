#define STB_IMAGE_IMPLEMENTATION
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "camera.h"
#include "cuPrintf.cu"
#include "curve.h"
#include "material.h"
// #include "mesh.h"
#include "object.h"
#include "object_list.h"
#include "plane.h"
#include "revsurface.h"
#include "scene.h"
#include "sphere.h"
#include "stb_image.h"
#include "texture.h"
#include "triangle.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
using namespace std;
#define DegreesToRadians(x) ((M_PI * x) / 180.0f)

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

static void check_cuda(cudaError_t result, char const *const func,
                       const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

Scene::Scene(const char *filename) {
    // initialize some reasonable default values
    objList = nullptr;
    camera = nullptr;
    num_materials = 0;
    materials = nullptr;
    current_material = nullptr;

    num_textures = 0;
    textures = nullptr;

    num_bumps = 0;
    bumps = nullptr;
    // parse the file
    assert(filename != nullptr);
    const char *ext = &filename[strlen(filename) - 4];

    if (strcmp(ext, ".txt") != 0) {
        printf("wrong file name extension\n");
        exit(0);
    }
    file = fopen(filename, "r");

    if (file == nullptr) {
        printf("cannot open scene file\n");
        exit(0);
    }
    parseFile();
    fclose(file);
    file = nullptr;
}

__global__ void freeScene(Camera **camera, Object **objList,
                          Material **materials, Texture **textures,
                          Bump **bumps, int num_materials, int num_textures,
                          int num_bumps) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *camera;
        delete *objList;
        for (int i = 0; i < num_materials; i++) delete materials[i];
        delete *materials;
        for (int i = 0; i < num_textures; i++) delete textures[i];
        delete *textures;
        for (int i = 0; i < num_bumps; i++) delete bumps[i];
        delete *bumps;
    }
}

Scene::~Scene() {
    // freeScene<<<1, 1>>>(camera, objList, materials, textures, bumps,
    //                     num_materials, num_textures, num_bumps);
    // for (int i = 0; i < d_pics.size(); ++i)
    //     checkCudaErrors(cudaFree(d_pics[i]));
}

// ====================================================================
// ====================================================================

void Scene::parseFile() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    while (getToken(token)) {
        if (!strcmp(token, "Camera")) {
            parseCamera();
        } else if (!strcmp(token, "Materials")) {
            parseMaterials();
        } else if (!strcmp(token, "Textures")) {
            parseTextures();
        } else if (!strcmp(token, "Bumps")) {
            parseBumps();
        } else if (!strcmp(token, "Objects")) {
            parseObjectList();
        } else {
            printf("Unknown token in parseFile: '%s'\n", token);
            exit(0);
        }
    }
}

// ====================================================================
// ====================================================================
__global__ void createCamera(Camera **camera, Vec3 *center, Vec3 *dir, Vec3 *up,
                             float angle, float focalLength, float aperture,
                             int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("New Camera at %d\n", camera);
        *camera = new Camera(*center, *dir, *up, angle, float(nx), float(ny),
                             aperture, focalLength);
    }
}

void Scene::parseCamera() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    Vec3 *center, *direction, *up;
    checkCudaErrors(cudaMallocManaged((void **)&center, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&direction, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&up, sizeof(Vec3)));
    float angle, focalLength = 1, aperture = 0;
    // read in the camera parameters
    getToken(token);
    assert(!strcmp(token, "{"));
    while (true) {
        getToken(token);
        if (!strcmp(token, "center")) {
            *center = readVec3();
        } else if (!strcmp(token, "direction")) {
            *direction = readVec3();
        } else if (!strcmp(token, "up")) {
            *up = readVec3();
        } else if (!strcmp(token, "angle")) {
            float angle_degrees = readFloat();
            angle = DegreesToRadians(angle_degrees);
        } else if (!strcmp(token, "width")) {
            width = readFloat();
        } else if (!strcmp(token, "height")) {
            height = readFloat();
        } else if (strcmp(token, "focalLength") == 0) {
            focalLength = readFloat();
        } else if (strcmp(token, "aperture") == 0) {
            aperture = readFloat();
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
    checkCudaErrors(cudaMallocManaged((void **)&camera, sizeof(Camera *)));
    createCamera<<<1, 1>>>(camera, center, direction, up, angle, focalLength,
                           aperture, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::parseMaterials() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numMaterials"));
    num_materials = readInt();
    checkCudaErrors(cudaMallocManaged((void **)&materials,
                                      num_materials * sizeof(Material *)));
    // read in the objects
    int count = 0;
    while (num_materials > count) {
        getToken(token);
        if (!strcmp(token, "Material")) {
            // printf("%d before: %d\n", count, materials[count]);
            parseMaterial(&materials[count]);
            // printf("%d after: %d\n", count, materials[count]);
        } else {
            printf("Unknown token in parseMaterial: '%s'\n", token);
            exit(0);
        }
        count++;
    }
    getToken(token);
    assert(!strcmp(token, "}"));
}

__global__ void createMaterial(Material **material, Texture *texture, Vec3 *a,
                               Vec3 *t, Vec3 *e, float refr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *material = new Material(*a, *t, *e, refr, texture);
        printf("New Material: %d\n", *material);
    }
}

void Scene::parseMaterial(Material **material) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    Texture *texture = nullptr;
    Vec3 *color, *emission, *type;
    checkCudaErrors(cudaMallocManaged((void **)&color, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&emission, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&type, sizeof(Vec3)));
    (*emission)[0] = (*emission)[1] = (*emission)[2] = 0;
    float refr = 1;
    int index;
    getToken(token);
    assert(!strcmp(token, "{"));
    while (true) {
        getToken(token);
        if (strcmp(token, "color") == 0) {
            *color = readVec3();
        } else if (strcmp(token, "specularColor") == 0) {
            readVec3();
        } else if (strcmp(token, "shininess") == 0) {
            readFloat();
        } else if (strcmp(token, "refr") == 0) {
            refr = readFloat();
        } else if (strcmp(token, "texture") == 0) {
            int index = readInt();
            assert(index >= 0 && index < num_textures);
            texture = textures[index];
        } else if (strcmp(token, "type") == 0) {
            *type = readVec3();
        } else if (strcmp(token, "emission") == 0) {
            *emission = readVec3();
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
    // cerr << *material << endl;
    createMaterial<<<1, 1>>>(material, texture, color, type, emission, refr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // cerr << *material << endl;
}

void Scene::parseTextures() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "numTextures"));
    num_textures = readInt();
if (num_textures){
checkCudaErrors(cudaMallocManaged((void **)&textures,
                                    num_textures * sizeof(Texture *)));
for (int i = 0; i < num_textures; ++i) parseTexture(&textures[i]);}
    getToken(token);
    assert(!strcmp(token, "}"));
}

__global__ void createTexture(Texture **texture, unsigned char *pic, int w,
                              int h, int c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *texture = new Texture(pic, w, h, c);
        printf("Texture file loaded. Size: %dx%dx%d\n", w, h, c);
    }
}

void Scene::parseTexture(Texture **texture) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    int w, h, c;
    unsigned char *pic = stbi_load(token, &w, &h, &c, 0);
    unsigned char *d_pic;
    checkCudaErrors(
        cudaMallocManaged((void **)&d_pic, w * h * c * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_pic, pic, w * h * c * sizeof(unsigned char),
                               cudaMemcpyHostToDevice));
    createTexture<<<1, 1>>>(texture, d_pic, w, h, c);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::parseBumps() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "numBumps"));
    num_bumps = readInt();
    checkCudaErrors(
        cudaMallocManaged((void **)&bumps, num_bumps * sizeof(Bump *)));
    for (int i = 0; i < num_bumps; ++i) parseBump(&bumps[i]);
    getToken(token);
    assert(!strcmp(token, "}"));
}

__global__ void createBump(Bump **bump, unsigned char *pic, int w, int h,
                           int c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *bump = new Bump(pic, w, h, c);
    }
}

void Scene::parseBump(Bump **bump) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    int w, h, c;
    unsigned char *pic = stbi_load(token, &w, &h, &c, 0);
    printf("Bump file: %s loaded. Size: %dx%dx%d\n", token, w, h, c);
    unsigned char *d_pic;
    checkCudaErrors(
        cudaMallocManaged((void **)&d_pic, w * h * c * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_pic, pic, w * h * c * sizeof(unsigned char),
                               cudaMemcpyHostToDevice));
    // d_pics.push_back(d_pic);
    createBump<<<1, 1>>>(bump, d_pic, w, h, c);
}

// ====================================================================
// ====================================================================

void Scene::parseObject(Object **obj, char token[MAX_PARSER_TOKEN_LENGTH]) {
    if (!strcmp(token, "Sphere")) {
        parseSphere(obj);
    } else if (!strcmp(token, "Plane")) {
        parsePlane(obj);
    } else if (!strcmp(token, "Triangle")) {
        parseTriangle(obj);
    }
    // else if (!strcmp(token, "TriangleMesh")) {
    //     parseTriangleMesh(obj);
    // }
    else if (!strcmp(token, "BezierCurve")) {
        parseBezierCurve(obj);
    } else if (!strcmp(token, "RevSurface")) {
        parseRevSurface(obj);
    } else {
        printf("Unknown token in parseObject: '%s'\n", token);
        exit(0);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

// ====================================================================
// ====================================================================
__global__ void createObjectList(Object **objList, Object **list, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *objList = new ObjectList(list, n);
    }
}

void Scene::parseObjectList() {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    // read in the number of objects
    getToken(token);
    assert(!strcmp(token, "numObjects"));
    int num_objects = readInt();
    checkCudaErrors(cudaMallocManaged((void **)&objList, sizeof(ObjectList *)));
    Object **d_list;
    checkCudaErrors(
        cudaMallocManaged((void **)&d_list, num_objects * sizeof(Object *)));
    // read in the objects
    int count = 0;
    while (num_objects > count) {
        getToken(token);
        printf("%d: %s\n", count, token);
        if (!strcmp(token, "MaterialIndex")) {
            int index = readInt();
            assert(index >= 0 && index < num_materials);
            current_material = materials[index];
        } else {
            parseObject(d_list + count, token);
            count++;
        }
    }
    getToken(token);
    assert(!strcmp(token, "}"));
    createObjectList<<<1, 1>>>(objList, d_list, num_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

// ====================================================================
// ====================================================================

__global__ void createSphere(Object **sphere, Vec3 *center, float r,
                             Material *material, Bump *bump) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Sphere\n");
        *sphere = new Sphere(*center, r, material, bump);
    }
}

void Scene::parseSphere(Object **sphere) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    Vec3 *center;
    checkCudaErrors(cudaMallocManaged((void **)&center, sizeof(Vec3)));
    float radius;
    Bump *bump = nullptr;
    while (true) {
        getToken(token);
        if (strcmp(token, "center") == 0) {
            *center = readVec3();
        } else if (strcmp(token, "radius") == 0) {
            radius = readFloat();
        } else if (strcmp(token, "bump") == 0) {
            int index = readInt();
            assert(index >= 0 && index < num_bumps);
            bump = bumps[index];
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
    assert(current_material != nullptr);
    createSphere<<<1, 1>>>(sphere, center, radius, current_material, bump);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createPlane(Object **plane, Vec3 *normal, float offset,
                            Material *material, Bump *bump) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *plane = new Plane(*normal, offset, material, bump);
    }
}

void Scene::parsePlane(Object **obj) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    Vec3 *normal;
    checkCudaErrors(cudaMallocManaged((void **)&normal, sizeof(Vec3)));
    float offset;
    Bump *bump = nullptr;
    while (true) {
        getToken(token);
        if (strcmp(token, "normal") == 0) {
            *normal = readVec3();
        } else if (strcmp(token, "offset") == 0) {
            offset = readFloat();
        } else if (strcmp(token, "bump") == 0) {
            int index = readInt();
            assert(index >= 0 && index < num_bumps);
            bump = bumps[index];
        } else {
            assert(!strcmp(token, "}"));
            break;
        }
    }
    assert(current_material != nullptr);
    createPlane<<<1, 1>>>(obj, normal, offset, current_material, bump);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createTriangle(Object **triangle, Vec3 *v0, Vec3 *v1, Vec3 *v2,
                               Material *material) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *triangle = new Triangle(*v0, *v1, *v2, material);
    }
}

void Scene::parseTriangle(Object **triangle) {
    Vec3 *v0, *v1, *v2;
    checkCudaErrors(cudaMallocManaged((void **)&v0, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&v1, sizeof(Vec3)));
    checkCudaErrors(cudaMallocManaged((void **)&v2, sizeof(Vec3)));
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "vertex0"));
    *v0 = readVec3();
    getToken(token);
    assert(!strcmp(token, "vertex1"));
    *v1 = readVec3();
    getToken(token);
    assert(!strcmp(token, "vertex2"));
    *v2 = readVec3();
    getToken(token);
    assert(!strcmp(token, "}"));
    assert(current_material != nullptr);
    createTriangle<<<1, 1>>>(triangle, v0, v1, v2, current_material);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

// void Scene::parseTriangleMesh(Object *mesh) {
//     char token[MAX_PARSER_TOKEN_LENGTH];
//     char filename[MAX_PARSER_TOKEN_LENGTH];
//     // get the filename
//     getToken(token);
//     assert(!strcmp(token, "{"));
//     getToken(token);
//     assert(!strcmp(token, "obj_file"));
//     getToken(filename);
//     getToken(token);
//     assert(!strcmp(token, "}"));
//     const char *ext = &filename[strlen(filename) - 4];
//     assert(!strcmp(ext, ".obj"));
//     Mesh *answer = new Mesh(filename, current_material);
//     return answer;
// }

__global__ void createBezierCurve(Object **curve, Vec3 *controls,
                                  int num_controls) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *curve = new BezierCurve(controls, num_controls);
    }
}

void Scene::parseBezierCurve(Object **curve) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "controls"));
    int num_controls = readInt();
    Vec3 *controls = new Vec3[num_controls];
    for (int i = 0; i < num_controls; ++i) {
        getToken(token);
        if (!strcmp(token, "[")) {
            controls[i] = readVec3();
            getToken(token);
            assert(!strcmp(token, "]"));
        } else if (!strcmp(token, "}")) {
            break;
        } else {
            printf("Incorrect format for BezierCurve!\n");
            exit(0);
        }
    }
    Vec3 *d_controls;
    checkCudaErrors(
        cudaMallocManaged((void **)&d_controls, num_controls * sizeof(Vec3)));
    checkCudaErrors(cudaMemcpy(d_controls, controls,
                               num_controls * sizeof(Vec3),
                               cudaMemcpyHostToDevice));
    createBezierCurve<<<1, 1>>>(curve, controls, num_controls);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createRevSurface(Object **surface, Object *profile,
                                 Material *material) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *surface = new RevSurface((Curve *)profile, material);
    }
}

void Scene::parseRevSurface(Object **surface) {
    char token[MAX_PARSER_TOKEN_LENGTH];
    getToken(token);
    assert(!strcmp(token, "{"));
    getToken(token);
    assert(!strcmp(token, "profile"));
    Object *profile;
    checkCudaErrors(cudaMallocManaged((void **)&profile, sizeof(Curve)));
    getToken(token);
    if (!strcmp(token, "BezierCurve")) {
        parseBezierCurve(&profile);
    } else {
        printf("Unknown profile type in parseRevSurface: '%s'\n", token);
        exit(0);
    }
    getToken(token);
    assert(!strcmp(token, "}"));
    createRevSurface<<<1, 1>>>(surface, profile, current_material);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

// ====================================================================
// ====================================================================

int Scene::getToken(char token[MAX_PARSER_TOKEN_LENGTH]) {
    // for simplicity, tokens must be separated by whitespace
    assert(file != nullptr);
    int success = fscanf(file, "%s ", token);
    if (success == EOF) {
        token[0] = '\0';
        return 0;
    }
    return 1;
}

Vec3 Scene::readVec3() {
    float x, y, z;
    int count = fscanf(file, "%f %f %f", &x, &y, &z);
    if (count != 3) {
        printf("Error trying to read 3 floats to make a Vec3\n");
        assert(0);
    }
    return Vec3(x, y, z);
}

float Scene::readFloat() {
    float answer;
    int count = fscanf(file, "%f", &answer);
    if (count != 1) {
        printf("Error trying to read 1 float\n");
        assert(0);
    }
    return answer;
}

int Scene::readInt() {
    int answer;
    int count = fscanf(file, "%d", &answer);
    if (count != 1) {
        printf("Error trying to read 1 int\n");
        assert(0);
    }
    return answer;
}
