from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.conv_auto_encoder import ConvAutoEncoder
import numpy as np
import cv2
import os
import json

EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_SIZE = (256, 256)
base_dataset = "subclass_400"
class_dir = ['tubular_adenoma', 'phyllodes_tumor', 'papillary_carcinoma',
             'mucinous_carcinoma', 'lobular_carcinoma', 'fibroadenoma',
             'ductal_carcinoma', 'adenosis']
type_dataset = ['val', 'train']

# make that checkpoint path dir are available
checkpoint_path = "checkpoint/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
print("[INFO] indexing file images BreakHis dataset...")
dataset_train = []
dataset_val = []
for type_set in type_dataset:
    for class_item in class_dir:
        cur_dir = os.path.join(base_dataset, type_set, class_item)
        for file in os.listdir(cur_dir):
            if type_set == 'train':
                dataset_train.append(os.path.join(cur_dir, file))
            else:
                dataset_val.append(os.path.join(cur_dir, file))

print("[INFO] load images BreakHis dataset...")
#  load images
train_images = []
val_images = []
for type_set in type_dataset:
    cur_dataset = dataset_train if type_set == 'train' else dataset_val
    for image_path in cur_dataset:
        if ".png" in image_path:
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            if type_set == 'train':
                train_images.append(image)
            else:
                val_images.append(image)

# normalization
print("[INFO] normalization...")
train_x = np.array(train_images).astype("float32") / 255.0
val_x = np.array(val_images).astype("float32") / 255.0

print("[INFO] building auto encoder...")
auto_encoder = ConvAutoEncoder.build(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_loss",
    verbose=1,
    mode='min',
    save_best_only=True)
auto_encoder.compile(loss="mse", optimizer=opt)
auto_encoder.summary()

# train the convolutional auto encoder
print("[INFO] training auto encoder...")
H = auto_encoder.fit(
    train_x, train_x,
    shuffle=True,
    validation_data=(val_x, val_x),
    epochs=EPOCHS,
    callbacks=[checkpoint_callback],
    batch_size=BS)

# save history training and model h5
with open('training_binary_sample_400.json', 'w') as f:
    json.dump(H.history, f)
auto_encoder.save('training_binary_sample_400.h5')
