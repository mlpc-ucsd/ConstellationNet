import numpy as np
from PIL import Image
import pickle
import os

# load the image
img_dir = 'raw-pacs/pacs_data/pacs_data/photo/dog/'
data = []
labels = []

for filename in os.listdir(img_dir):
    if filename.endswith((".jpg", ".png")):
        data.append(np.asarray(Image.open(img_dir + filename)))
        labels.append(1)

# Create a dict for pickling
pickle_dict = {
    "data": data,
    "labels": labels,
}

pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_test.pickle", "wb"))
pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_train.pickle", "wb"))
pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_val.pickle", "wb"))
