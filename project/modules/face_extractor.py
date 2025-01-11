import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceExtractor:
    MODEL_PATH = '../blaze_face_short_range.tflite'
    def __init__(self,*, mode=None, path=None):
         self.detector = vision.FaceDetector.create_from_options(
              vision.FaceDetectorOptions(base_options=python.BaseOptions(model_asset_path=path or self.MODEL_PATH)))
         self.mode = mode
           

    def extract_bounding_boxes(self, results):

        detections = []
        for det in results:
            detections.append(self.extract_detections(det))
        
        return detections


    def extract_detections(self, det):
            
            x = det.bounding_box.origin_x
            y = det.bounding_box.origin_y
            w = det.bounding_box.width
            h = det.bounding_box.height
    
            return x, y, w, h
    
    
    def read_from_file(self, path): return mp.Image.create_from_file(path)
    def adapt_image(self, img, *, gray=False): 
        return mp.Image(image_format=(mp.ImageFormat.GRAY8 if gray else mp.ImageFormat.SRGB), data=img)
   
    def __call__(self, img): return self.extract_bounding_boxes(self.detector.detect( img if self.mode == 'mp' else self.adapt_image(img)).detections)