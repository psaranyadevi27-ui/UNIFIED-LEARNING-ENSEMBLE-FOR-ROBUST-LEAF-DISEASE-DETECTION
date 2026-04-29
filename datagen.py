import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from feature_extraction import feature_extraction
from save_load import save
from sklearn.model_selection import train_test_split


def apply_log(preprocessed_image, alpha=0.8):
    # Convert the image to grayscale
    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), alpha)

    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    _, binary_mask = cv2.threshold(np.abs(laplacian), 15, 29, cv2.THRESH_BINARY)

    return binary_mask


def datagen():
    BaseDir = './Dataset/PlantVillage/'
    imgResult = './Pictorial Results/'
    FolderDir = os.listdir(BaseDir)
    n = 0
    label = []
    features = []
    for folder in FolderDir:
        imageDir = os.listdir(BaseDir + folder)
        for image in imageDir:
            imgs = cv2.imread(BaseDir + folder + '/' + image)
            if imgs is None:
                continue
            # imgs = cv2.imread('./Dataset/PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus/0a9e37a2-95d5-4af7-95a1-2cbb67074452___YLCV_NREC 2291.JPG')

            # preprocessing
            # Image Cleaning - Non-Local Means
            cleaned_image = cv2.fastNlMeansDenoisingColored(imgs, None, 10, 10, 7, 15)
            # Image Resizing
            Resized_Image = cv2.resize(cleaned_image, (250, 250))

            # Image Normalization
            normalized_image = cv2.normalize(Resized_Image, None, 0, 255, norm_type=cv2.NORM_MINMAX)

            # Apply Laplacian of Gaussian
            SegImage = apply_log(normalized_image)

            # plt.figure(figsize=(10, 8))
            #
            # plt.subplot(2, 3, 1)
            # plt.imshow(imgs, cmap='gray')
            # plt.title('Original Image')
            #
            # plt.subplot(2, 3, 2)
            # plt.imshow(cleaned_image, cmap='gray')
            # plt.title('Cleaned Image')
            #
            # plt.subplot(2, 3, 3)
            # plt.imshow(Resized_Image, cmap='gray')
            # plt.title('Resized Image')
            #
            # plt.subplot(2, 3, 4)
            # plt.imshow(normalized_image, cmap='gray')
            # plt.title('Normalized Image')
            #
            # plt.subplot(2, 3, 5)
            # plt.imshow(SegImage)
            # plt.title('Segmented Image')
            #
            # plt.savefig(imgResult + 'Image.jpg')
            # plt.show()

            feature = feature_extraction(SegImage)
            label.append(n)
            features.append(feature)
        n += 1

    features = np.array(features)

    features = abs(features)
    features = features / np.max(features, axis=0)
    features = np.nan_to_num(features)

    label = np.array(label)
    learning_rate = [0.7, 0.8]  # train size
    for learn_rate in learning_rate:
        x_train, x_test, y_train, y_test = train_test_split(features, label, train_size=learn_rate)
        save('x_train_' + str(int(learn_rate * 100)), x_train)
        save('x_test_' + str(int(learn_rate * 100)), x_test)
        save('y_train_' + str(int(learn_rate * 100)), y_train)
        save('y_test_' + str(int(learn_rate * 100)), y_test)

datagen()