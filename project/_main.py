import sys

from modules.Comparator import *
from modules.face_extractor import *
from modules.img_transforms import *
from modules.Model import *
from commons.imgs_mean_std import *

import torch
import cv2
import torchvision.transforms as transforms
import base64

# consts
THRESHOLD = 2.0
MODEL_PATH = './model.pth'


# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# face_extractor
detector = FaceExtractor(mode=None, path='blaze_face_short_range.tflite')


# normalizer
normalizer = normalize(220, MEAN, STD)

# functions

tensor2str = lambda embs : base64.b64encode(','.join(str(elm) for elm in embs.numpy()[0]).encode('utf-8')).decode('utf-8')
str2tensor = lambda embs : torch.Tensor([[float(elm) for elm in base64.b64decode(embs).decode('utf-8').strip('[]').split(',')]])


def calc_embeddings(path): 

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    detections = detector(img)
    x,y,w,h = detections[0]
    face = img[y:y+h,x:x+w]

    norm_face_tensor = normalizer(face).to(device).unsqueeze(0)

    with torch.no_grad(): embs = model(norm_face_tensor)

    return embs.cpu()


def exit():
    sys.exit(0)

if __name__ == '__main__':        

    print('\\#')

    if sys.argv[1] == 'calc':
        print(tensor2str(calc_embeddings( sys.argv[2])))
        exit()

    if sys.argv[1] == 'comp':
        dist = dst(calc_embeddings(sys.argv[2]), str2tensor(sys.argv[3])).item()
        print(f'match ({dist})' if dist < THRESHOLD else f'mismatch ({dist})')
        exit()

    else:
        print('calc image_path\tto calc embeddings')
        print('comp image_path embeding\tto calc if an embedding is from a person')
        exit()