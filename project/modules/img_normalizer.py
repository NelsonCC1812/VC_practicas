from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

normalize = lambda image_size:  transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size), antialias=True),
    transforms.functional.equalize()
    #transforms.ToTensor()
])