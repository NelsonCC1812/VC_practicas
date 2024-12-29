import cv2
import numpy as np
import mediapipe as mp
from torchvision import transforms

# --- 
import sys; sys.path.append('../')
from commons.imageutils import *


# facemesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


# def face_aligner(img):

#     img = tensor2numpy(img)
#     plt.imshow(img)
    
#     res = facemesh.process(img)

#     if res.multi_face_landmarks:
#         for facelm in res.multi_face_landmarks:

#             leye, reye = facelm.landmark[33], facelm.landmark[263]
#             leye, reye = (int(leye.x * img.shape[1]), int(leye.y * img.shape[0])), (int(reye.x * img.shape[1]), int(reye.y * img.shape[0]))

#             degress = np.degrees(np.arctan2((reye[1]-leye[1]),(reye[0]-leye[0])))
#             center = ((leye[0] + reye[0]) // 2, (leye[1] + reye[1]) // 2)
#             rot_mat = cv2.getRotationMatrix2D(center, degress, 1.0)
            
#             return cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))


# normalize = lambda image_size: lambda img: transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((image_size, image_size), antialias=True),
#     transforms.functional.equalize,
#     transforms.ToTensor()
# ])(face_aligner(img))

normalize = lambda image_size, mean=0, std=1: transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size), antialias=True),
    transforms.functional.equalize,
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


data_augmentation = lambda: transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(p=.8),
    transforms.RandomVerticalFlip(p=.8),
    transforms.ColorJitter(brightness=.3, contrast=.1),
    transforms.ToTensor()
])