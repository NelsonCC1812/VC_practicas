import matplotlib.pyplot as plt

def show_img(img, title=None, gray=False):
    plt.imshow(img.permute(1,2,0), cmap='gray' if gray else None)
    plt.axis('off')
    if title: plt.title(title)

tensor2numpy_gray = lambda img: (img.permute(1,2,0).numpy()*255).astype('uint8')
tensor2numpy = lambda img: (img.numpy().transpose(1,2,0)*255).astype('uint8')