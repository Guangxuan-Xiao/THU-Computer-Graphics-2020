#ifndef SCENE_H
#define SCENE_H

#include <cassert>

#include "vec3.h"
#define MAX_PARSER_TOKEN_LENGTH 1024
class Camera;
class Material;
class Object;
class ObjectList;
class Sphere;
class Plane;
class Triangle;
class Mesh;
class Curve;
class RevSurface;
struct Bump;
struct Texture;
// using std::vector;
class Scene {
   public:
    Scene(const char *filename);
    ~Scene();
    FILE *file;
    float width, height;
    // vector<unsigned char *> d_pics;
    // Following objects are on device.
    Camera **camera;

    int num_materials;
    Material **materials;
    Material *current_material;

    int num_textures;
    Texture **textures;
    int num_bumps;
    Bump **bumps;

    Object **objList;

   private:
    void parseFile();
    void parseCamera();
    void parseMaterials();
    void parseMaterial(Material **material);
    void parseTextures();
    void parseTexture(Texture **texutre);
    void parseBumps();
    void parseBump(Bump **bump);
    void parseObject(Object **obj, char token[MAX_PARSER_TOKEN_LENGTH]);
    void parseObjectList();
    void parseSphere(Object **sphere);
    void parsePlane(Object **plane);
    void parseTriangle(Object **triangle);
    // void parseTriangleMesh(Object *mesh);
    void parseBezierCurve(Object **curve);
    void parseRevSurface(Object **revSurface);

    int getToken(char token[MAX_PARSER_TOKEN_LENGTH]);

    Vec3 readVec3();

    float readFloat();
    int readInt();
};
#endif
