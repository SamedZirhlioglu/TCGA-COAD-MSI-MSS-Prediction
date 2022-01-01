import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

DATASET = r"E:\finished_projects\machine_learning\samed_enes\preprocessed_data"
TEST = DATASET + "\\test"
TRAIN = DATASET + "\\train"
VALIDATION = DATASET + "\\val"

def start_timer(name):
    print(name + " Data Importing Started")
    return time.time()

def calculate_time(start_time, name):
    print(name + " Data Importing Finished ({:.2f} seconds)".format(round((time.time() - start_time), 2)))

def import_images(folder_path, target_folder, extension=".jpg"):
    paths = []
    images = []
    for file in os.listdir(folder_path + "\\" + target_folder):
        if file.endswith(extension):
            image_path = os.path.join(folder_path, target_folder, file)
            images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            paths.append(image_path)
    
    return images, paths

def show_images(image_1, image_2, header_1="Original", header_2="Edited"):
    plt.subplot(121)    
    plt.imshow(image_1)
    plt.title(header_1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(image_2)
    plt.title(header_2)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def process(dataset, paths):
    images = []

    for image in dataset:
        
        blured_image = cv2.GaussianBlur(image, (5, 5), 0)
        ret, segmented_image = cv2.threshold(blured_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #show_images(image, blured_image)

        ones = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, ones, iterations=2)
        background = cv2.dilate(morph, ones, iterations=3)
        #show_images(blured_image, background)
        
        d_trans = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
        ret, foreground = cv2.threshold(d_trans, 0.7 * d_trans.max(), 255, 0)
        
        foreground = np.uint8(foreground)
        #show_images(background, cv2.subtract(background, foreground))
        
        images.append(cv2.subtract(background, foreground))

    for i in range(len(dataset)):
        cv2.imwrite(paths[i], images[i])

def main():
    start_time = start_timer("Validation MSI")
    images, paths = import_images(VALIDATION, "MSI")
    calculate_time(start_time, "Validation MSI")
    process(images, paths)

    start_time = start_timer("Validation MSS")
    images, paths = import_images(VALIDATION, "MSS")
    calculate_time(start_time, "Validation MSS")
    process(images, paths)
    
    start_time = start_timer("Test MSI")
    images, paths = import_images(TEST, "MSI")
    calculate_time(start_time, "Test MSI")
    process(images, paths)

    start_time = start_timer("Test MSS")
    images, paths = import_images(TEST, "MSS")
    calculate_time(start_time, "Test MSS")
    process(images, paths)
    
    start_time = start_timer("Train MSI")
    images, paths = import_images(TRAIN, "MSI")
    calculate_time(start_time, "Train MSI")
    process(images, paths)

    start_time = start_timer("Train MSS")
    images, paths = import_images(TRAIN, "MSS")
    calculate_time(start_time, "Train MSS")
    process(images, paths)
    

main()