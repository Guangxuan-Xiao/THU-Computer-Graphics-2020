#ifndef TEXTURE_H
#define TEXTURE_H
#include <string>

#include <vecmath.h>
using std::string;
struct Texture {  // 纹理
    unsigned char *pic;
    int w, h, c;
    Texture(const char *textureFile);
    Vector3f getColor(float u, float v) const;
    Vector3f getColor(int idx) const;
    Vector3f getColor(int u, int v) const;
    float getDisturb(float u, float v, Vector2f &grad) const;
    inline int getIdx(float u, float v) const;
    inline float getGray(int idx) const;
};

#endif  // !TEXTURE_H