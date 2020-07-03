import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
arg = parser.parse_args()
i = cv2.imread(arg.input)
cv2.imwrite(arg.output, i)
