# Literature:
# https://learnopencv.com/eigenface-using-opencv-c-python/
# https://github.com/informramiz/opencv-face-recognition-python/blob/master/README.md
# https://github.com/JDAI-CV/faceX-Zoo
# https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/
# http://rogerioferis.com/publications/FerisWAICV00.pdf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.spatial.distance import euclidean
import face_recognition
from pprint import pprint


def createDataMatrix(images):
    sz = images[0][0].shape
    print(f"image size shape {sz}")
    flatten_size = sz[0] * sz[1] * sz[2]
    print(f"image size shape flatten size = {flatten_size}")

    data = np.zeros((len(images), flatten_size), dtype=np.float32)
    for i in range(0, len(images)):
        data[i, :] = images[i][0].flatten()

    return data


# Read images from the directory
def readImages(path, train_files):
    images = []
    names = []

    for filePath in tqdm(np.array(train_files)[:, 0]):
        # Add to array of images
        imagePath = os.path.join(path, filePath)
        names.append(filePath)
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        try:
            # im = crop_right_eye(im)
            im = FUNC(im)
        except:
            continue

        if im is None:
            print("image:{} not read properly".format(imagePath))
        else:
            # scaling in range 0..1
            im = np.float32(im) / 255.0

            # Add image to list
            images.append((im, filePath))

    numImages = len(images)
    # Exit if no image found
    if numImages == 0:
        print("No images found")
        exit()

    print(str(numImages) + " files read.")
    return images


def read_labels(path):
    with open(path, 'r', encoding='utf-8') as reader:
        data = reader.readlines()
    data = [sample.strip().split(" ") for sample in data]
    data = {key: value for (key, value) in data}

    return data


def read_test_labels(path="../test_images.npy"):
    data = np.load(path)
    return data


def read_train_files(path="../sample_1k.txt"):
    with open(path, 'r', encoding='utf-8') as reader:
        data = reader.readlines()
    data = [sample.strip().split(",")[1:] for sample in data]
    return data


def crop_left_eye(im):
    face_landmarks_list = face_recognition.face_landmarks(im)

    left_eye_top_left = face_landmarks_list[0]["left_eye"][0]
    left_eye_bottom_right = face_landmarks_list[0]["left_eye"][3]

    left_eye_crop = im[left_eye_top_left[1] - 10:left_eye_bottom_right[1] + 10,
                    left_eye_top_left[0] - 10:left_eye_bottom_right[0] + 10]
    left_eye_crop = cv2.resize(left_eye_crop, (32, 32))

    return left_eye_crop


def crop_right_eye(im):
    face_landmarks_list = face_recognition.face_landmarks(im)

    right_eye_bottom_right = face_landmarks_list[0]["right_eye"][3]
    right_eye_top_left = face_landmarks_list[0]["right_eye"][0]

    right_eye_crop = im[right_eye_top_left[1] - 10:right_eye_bottom_right[1] + 10,
                     right_eye_top_left[0] - 10:right_eye_bottom_right[0] + 10]
    right_eye_crop = cv2.resize(right_eye_crop, (32, 32))

    return right_eye_crop


FUNC = crop_left_eye


def test():
    """


    # nose_left = face_landmarks_list[0]["nose_tip"][0]
    # nose_right = face_landmarks_list[0]["nose_tip"][4]
    #
    # lips_left = face_landmarks_list[0]["top_lip"][0]
    # lips_right = face_landmarks_list[0]["top_lip"][6]

    # im = cv2.circle(im, face_landmarks_list[0]["top_lip"][6], radius=5, color=(0, 0, 255), thickness=-1)

    left_eye_crop = im[left_eye_top_left[1] - 10:left_eye_bottom_right[1] + 10,
                    left_eye_top_left[0] - 10:left_eye_bottom_right[0] + 10]
    left_eye_crop = cv2.resize(left_eye_crop, (32, 32), cv2.INTER_LANCZOS4)
    # TODO: add more crops more eigenfeatures
    # right_eye_crop = im[right_eye_top_left[1] - 10:right_eye_bottom_right[1] + 10,
    #                 right_eye_top_left[0] - 10:right_eye_bottom_right[0] + 10]
    # right_eye_crop = cv2.resize(left_eye_crop, (32, 32), cv2.INTER_LANCZOS4)
    """
    pass


def main():
    identity_labels = read_labels("data/identity_CelebA.txt")

    train_files = read_train_files()

    test_labels = read_test_labels()

    # Number of EigenFaces
    NUM_EIGEN_FACES = 512

    # Directory containing images
    dirName = "data/img_align_celeba"

    images = readImages(dirName, train_files)

    # Size of images
    sz = images[0][0].shape

    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("DONE")
    print(f"mean shape {mean.shape}")
    print(f"eigenVectors shape {eigenVectors.shape}")

    counter_test_skipped = 0
    distance = []
    for test in tqdm(test_labels):

        im = cv2.imread(
            f"eigenfeatures/data/img_align_celeba/{test}",
            cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        try:
            im = FUNC(im)
        except:
            print("test image keypoints not found")
            counter_test_skipped += 1
            continue

        im = np.float32(im) / 255.0

        weight_vector = np.matmul(eigenVectors, (im.flatten() - mean).reshape(mean.shape[1], 1))

        for B in trange(0, len(images)):

            metric = euclidean(np.matmul(eigenVectors, (images[B][0].flatten() - mean).reshape(mean.shape[1], 1)),
                               weight_vector)

            if identity_labels[images[B][1]] == identity_labels[test]:
                distance.append(metric)

    res = np.array(distance)
    print(f"mean = {round(res.mean(), 2)}")
    print(f"std = {round(res.std(), 2)}")
    print(f"NUM_EIGEN_FACES = {NUM_EIGEN_FACES}")
    print(f"counter_test_skipped = {counter_test_skipped}")

    exit()

    averageFace = mean.reshape(sz)

    eigenFaces = []

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    new_face = averageFace + 100 * eigenFaces[2]
    # new_face = cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB)
    plt.imshow(np.hstack((averageFace, new_face)))
    plt.show()


# TODO: deeplearning
# https://github.com/JDAI-CV/faceX-Zoo
# https://arxiv.org/pdf/2101.04407.pdf
# https://arxiv.org/pdf/1811.00116.pdf


if __name__ == '__main__':
    main()
