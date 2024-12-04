from collections import namedtuple

DatasetEntry = namedtuple("DatasetEntry", "info data")

DATASETS_PATHS = namedtuple("DatasetInfoPath", "originals processed faces norm_faces")(
    DatasetEntry("data/dataset/selfie_Id.csv", "data/dataset/Selfies ID Images dataset"),
    DatasetEntry("data/dataset/dataset.csv", "data/dataset/Selfies ID Images dataset"),
    DatasetEntry("data/face_images", "data/face_images"),
    DatasetEntry("data/normalized_face_images", "data/normalized_face_images")
    )