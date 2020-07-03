from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--r", type=str)
parser.add_argument("--g", type=str)
parser.add_argument("--b", type=str)
parser.add_argument("--out", type=str)
args = parser.parse_args()
rImg, gImg, bImg = Image.open(args.r), Image.open(args.g), Image.open(args.b)
r, _, _ = rImg.split()
_, g, _ = gImg.split()
_, _, b = bImg.split()
outImg = Image.merge('RGB', (r, g, b))
outImg.save(args.out)
