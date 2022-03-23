import numpy as np
from PIL import Image
import pickle

# load the image
image = Image.open('raw-pacs/pacs_data/pacs_data/photo/dog/056_0001.jpg')
# convert image to numpy array
picledict = {
    "data": [np.asarray(image), np.asarray(image)],
    "labels": [1, 2],
}

pickle.dump(picledict, open("materials/cifar-fs/CIFAR_FS_test.pickle", "wb"))
pickle.dump(picledict, open("materials/cifar-fs/CIFAR_FS_train.pickle", "wb"))
pickle.dump(picledict, open("materials/cifar-fs/CIFAR_FS_val.pickle", "wb"))
