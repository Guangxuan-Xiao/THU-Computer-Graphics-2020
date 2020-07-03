with open("bunny_1k.obj") as fin:
    lines = fin.readlines()


def transform(v):
    v[1] = str(5 * (float(v[1]) + 0.03))
    v[2] = str(5 * (float(v[2]) - 0.0666))
    v[3] = str(5 * float(v[3]))


for idx, line in enumerate(lines):
    if (line[0] == "v"):
        v = line.split()
        transform(v)

with open("bunny_1k_transformed.obj", "w+") as fout:
    fout.writelines(lines)
