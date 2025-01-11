import sys

from modules.Comparator import *
from modules.face_extractor import *
from modules.img_transforms import *
from modules.Model import *

import torch
import cv2

# consts
THRESHOLD = 2.5
MODEL_PATH = './model.pth'

# model
model = Model()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# face_extractor
detector = FaceExtractor()

# function
def calc_embeddings(path): 

    detections = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    detections = 


if __name__ == '__main__':

    if sys.argv[0] == 'calc':
        print(calc_embeddings( sys.argv[1]))
        sys.exit(0)

    if sys.argv[0] == 'comp':
        dist = dst(calc_embeddings(sys.argv[1]), sys.argv[2])
        print(f'match ({dist})' if dist < THRESHOLD else f'mismatch {dist}')
        sys.exit(0)

    else:
        print('calc image_path\tto calc embeddings')
        print('comp image_path embeding\tto calc if an embedding is from a person')
        sys.exit(0)