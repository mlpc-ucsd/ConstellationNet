import numpy as np
from PIL import Image
import pickle
import os

img_dir = 'raw-pacs/pacs_data/pacs_data/photo/'
classes_dict = {
    "dog": 1,
    "elephant": 2,
    "giraffe": 3,
    "guitar": 4,
    "horse": 5,
    "house": 6,
    "person": 7,
}
data = []
labels = []

print("Start pickling classes")
for class_ in classes_dict:
    for filename in os.listdir(img_dir + class_):
        if filename.endswith((".jpg", ".png")):
            data.append(np.asarray(Image.open(img_dir + class_ + "/" + filename)))
            labels.append(classes_dict[class_])

# Create a dict for pickling
pickle_dict = {
    "data": data,
    "labels": labels,
}

pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_test.pickle", "wb"))
pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_train.pickle", "wb"))
pickle.dump(pickle_dict, open("materials/cifar-fs/CIFAR_FS_val.pickle", "wb"))
print("Pickling finished")
