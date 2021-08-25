import numpy as np
import os
import matplotlib.pyplot as plt
import json
import cv2
from utils.retrieval import perform_search
from utils.conv_auto_encoder import ConvAutoEncoder
from tensorflow.keras.models import Model

base_dataset = "binary_scenario"
magnification = "400X"
class_dir = ['benign', 'malignant']
IMAGE_SIZE = (256, 256)
print("[INFO] indexing file images BreakHis dataset...")

if __name__ == "__main__":
    # indexing file images
    dataset = []
    for class_item in class_dir:
        cur_dir = os.path.join(base_dataset, 'test', magnification, class_item)
        for file in os.listdir(cur_dir):
            dataset.append(os.path.join(cur_dir, file))

    dataset_train = []
    for class_item in class_dir:
        cur_dir = os.path.join(base_dataset, 'train', magnification, class_item)
        for file in os.listdir(cur_dir):
            dataset_train.append(os.path.join(cur_dir, file))

    images_train = []
    for image_path in dataset_train:
        if ".png" in image_path:
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            images_train.append(image)

    print("[INFO] load images BreakHis dataset...")
    #  load images
    images = []
    for image_path in dataset:
        if ".png" in image_path:
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)

    # normalization
    print("[INFO] normalization...")
    test_x = np.array(images).astype("float32") / 255.0

    # 400X
    auto_encoder_400 = ConvAutoEncoder.build(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    auto_encoder_400.load_weights("checkpoint/cp.ckpt")
    with open('training_binary_sample_400.json') as f:
        training_indexed_400 = json.load(f)
    encoder_400 = Model(inputs=auto_encoder_400.input,
                        outputs=auto_encoder_400.get_layer("encoded").output)

    test_sample = test_x[10, :, :, :].reshape(1, 256, 256, 3)

    features_retrieved_400 = encoder_400.predict(test_sample)

    query_indexes = list(range(0, test_x.shape[0]))
    label_builder = list(np.unique(training_indexed_400["labels"]))
    class_builder = {label_unique: [] for label_unique in label_builder}

    plt.figure(figsize=(6, 6))
    plt.imshow(test_x[10])
    plt.show()

    queryFeatures = features_retrieved_400[0]
    results = perform_search(queryFeatures, training_indexed_400, max_results=5)
    labels_ret = [training_indexed_400["labels"][r[1]] for r in results]

    i = 0
    f, axs = plt.subplots(1, 5)
    f.set_figheight(20)
    f.set_figwidth(30)
    for label in labels_ret:
        axs[i].imshow(images_train[results[i][1]])
        axs[i].set_xlabel(label.upper())
        i += 1
    plt.show()
