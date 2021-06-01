import numpy as np
from PIL import Image

# resize the image to desired size
def image_resize_cum_preprocess(img_path, size):
    #load the image, convert to RGB, if needed and resize and finally convert it into numpy array.
    img_arr = np.asarray(Image.open(img_path).convert("RGB").resize(size)).astype('float32')
    # standardize pixel values across channels (global)
    mean, std = img_arr.mean(), img_arr.std()
    img_arr = (img_arr - mean) / std
    return np.expand_dims(img_arr, axis = 0)

def compute_dist(a,b):
    return np.sum(np.square(a-b))


def split_text(s): 
    s = s.split(".")[0].split("_")
    while True:
        s[-1] = s[-1][:-1]
        if s[-1].isalpha():
            break
    return "_".join(s)