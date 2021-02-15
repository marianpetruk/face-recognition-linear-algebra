import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.spatial.distance import euclidean


# import sys

def createDataMatrix(images):
    print("Creating data matrix", end=" ... ")
    ''' 
    Allocate space for all images in one data matrix. 
        The size of the data matrix is
        ( w  * h  * 3, numImages )

        where,

        w = width of an image in the dataset.
        h = height of an image in the dataset.
        3 is for the 3 color channels.
        '''

    numImages = len(images)
    print(f"numImages = {numImages}")
    sz = images[0][0].shape  # 218, 178, 3
    print(f"image size shape {sz}")
    print(f"image size shape flatten size {sz[0] * sz[1] * sz[2]}")

    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i][0].flatten()
        data[i, :] = image

    print("DONE in createDataMatrix")
    return data


# Read images from the directory
def readImages(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    images = []
    # List all files in the directory and read points from text files one by one
    for filePath in tqdm(sorted(os.listdir(path))):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg", ".png"]:

            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


            if im is None:
                print("image:{} not read properly".format(imagePath))
            else:
                # print(im.dtype)
                # exit()

                # TODO: why we do this?
                # Convert image to floating point
                im = np.float32(im) / 255.0


                # Add image to list
                images.append((im, filePath))

                # Flip image
                # If one wants to increase the dataset and make it symmetric
                # > 0 for flipping around the y-axis (horizontal flipping);
                # imFlip = cv2.flip(im, 1);
                # Append flipped image
                # images.append((imFlip, filePath))
            # plt.imshow(imFlip)
            # plt.show()
            # exit()


        if len(images) == 500:
            break

    numImages = len(images)
    # Exit if no image found
    if numImages == 0:
        print("No images found")
        exit()

    print(str(numImages) + " files read.")
    return images


# # Add the weighted eigen faces to the mean face
# def createNewFace(*args):
#     # Start with the mean image
#     output = averageFace
#
#     # Add the eigen faces with the weights
#     for i in xrange(0, NUM_EIGEN_FACES):
#         '''
#         OpenCV does not allow slider values to be negative.
#         So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
#         '''
#         sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
#         weight = sliderValues[i] - MAX_SLIDER_VALUE/2
#         output = np.add(output, eigenFaces[i] * weight)
#
#     # Display Result at 2x size
#     output = cv2.resize(output, (0,0), fx=2, fy=2)
#     cv2.imshow("Result", output)
#
# def resetSliderValues(*args):
#     for i in xrange(0, NUM_EIGEN_FACES):
#         cv2.setTrackbarPos("Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2);
#     createNewFace()

def read_labels(path):
    with open(path, 'r', encoding='utf-8') as reader:
        data = reader.readlines()
    data = [sample.strip().split(" ") for sample in data]
    data = {key: value for (key, value) in data}

    return data


def main():
    identity_labels = read_labels("data/identity_CelebA.txt")

    # Number of EigenFaces
    NUM_EIGEN_FACES = 512

    # Directory containing images
    dirName = "data/img_align_celeba"

    # The directory contains images that are aligned.
    #  We add these images to a list ( or vector ).
    #  We also flip the images vertically and add them to the list.
    #  Because the mirror image of a valid facial image, we just doubled the size of our dataset and made it symmetric at that same time.
    # Read images
    images = readImages(dirName)

    # Size of images
    sz = images[0][0].shape

    #  Each row of the data matrix is one image. Let’s look into the createDataMatrix function
    # Create data matrix for PCA.
    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("DONE")
    print(f"mean shape {mean.shape}")
    print(f"eigenVectors shape {eigenVectors.shape}")

    # A = 403 # 0 - 71.15271759033203
    # 003415 - 130.19
    A = 0
    weight_vector = np.matmul(eigenVectors, (images[A][0].flatten() - mean).reshape(mean.shape[1], 1))
    im = cv2.imread("/home/administrator/Documents/UCU/Study/Linear_algebra/final_project/repo/eigenfeatures/data/img_align_celeba/004390.jpg", cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.float32(im) / 255.0
    # TODO: compare number of eigenfaces with accuracy
    # TODO: compare number of fed images with accuracy

    metric = euclidean(np.matmul(eigenVectors, (im.flatten() - mean).reshape(mean.shape[1], 1)),
                       weight_vector)
    print(metric)
    exit()
    for B in trange(0, 405):

        # print((images[0].flatten() - mean).reshape(mean.shape[1], 1).shape)
        # exit()
        # print(f"face A = {identity_labels[images[A][1]]}")
        # print(weight_vector.shape)

        # print(np.matmul(eigenVectors, (images[1].flatten() - mean).reshape(mean.shape[1], 1)))
        # exit()

        # mse = ((A - B) ** 2).mean(axis=ax)
        # linalg.norm(a - b)

        # print(f"face B = {identity_labels[images[B][1]]}")

        metric = euclidean(np.matmul(eigenVectors, (images[B][0].flatten() - mean).reshape(mean.shape[1], 1)), weight_vector)
        print(metric)
        if metric < 75:
            print(f"face B = {identity_labels[images[B][1]]}")
            print(f"similar to A = {A}")

    exit()


    averageFace = mean.reshape(sz)

    eigenFaces = []

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    # Create window for displaying Mean Face
    # cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

    # Display result at 2x size
    # output = cv2.resize(averageFace, (0, 0), fx=2, fy=2)
    # output = cv2.cvtColor(averageFace, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Result", output)

    new_face = averageFace + 100 * eigenFaces[2]
    # new_face = cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB)
    plt.imshow(np.hstack((averageFace, new_face)))
    plt.show()

    # Create Window for trackbars
    # cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    # sliderValues = []

    # # Create Trackbars
    # for i in range(0, NUM_EIGEN_FACES):
    #     sliderValues.append(MAX_SLIDER_VALUE / 2)
    #     cv2.createTrackbar("Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE / 2, MAX_SLIDER_VALUE, createNewFace)
    #
    # # You can reset the sliders by clicking on the mean image.
    # cv2.setMouseCallback("Result", resetSliderValues);

    # print('''Usage:
    # 	Change the weights using the sliders
    # 	Click on the result window to reset sliders
    # 	Hit ESC to terminate program.''')

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# TODO: implement and compare cv2 default eigenface https://youtu.be/myKXW6SKLzY

#TODO: deeplearning
# https://github.com/JDAI-CV/faceX-Zoo
# https://arxiv.org/pdf/2101.04407.pdf
# https://arxiv.org/pdf/1811.00116.pdf




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
