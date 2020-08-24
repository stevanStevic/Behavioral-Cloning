import cv2
from matplotlib.pyplot import imread
from random import seed
from random import randint
from random import random

# seed random number generator
seed(1)

def image_preprocessing(img):
    # Remove unnecessary parts of the image
    img = img[61:137,:,:]

    img = hsv_conversion(img)

    img = gaussian_blur(img)

    # This is preffered size for neural network used
    img = cv2.resize(img, (200, 66))

    return img

def hsv_conversion(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

def read_one_image(absolute_path):
    filename = absolute_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    
    image = imread(current_path)

    return image

def flip_image(img):
    flip_around_y_axis = 1
    return cv2.flip(img, flip_around_y_axis)

def should_augment(augment):
    value = randint(0, 2)

    ret_val = (value == 1) and augment

    return ret_val

def read_data(batch_samples, augment):
    augment_cnt = 0
    images = []
    angles = []

    corner_images_angle_correct = 0.25

    for line in batch_samples:
        angle = float(line[3])

        # Center image
        image = read_one_image(line[0])
        image = image_preprocessing(image)

        if should_augment(augment):
            image = flip_image(image)
            angle = -angle  # Flip the steering controls

        images.append(image)
        angles.append(angle)

        # Left image
        image = read_one_image(line[1])
        image = image_preprocessing(image)

        angle_left = angle + corner_images_angle_correct
        if should_augment(augment):
            image = flip_image(image)
            angle_left = -angle_left

        images.append(image)
        angles.append(angle_left)

        # Right image
        image = read_one_image(line[2])
        image = image_preprocessing(image)

        angle_right = angle - corner_images_angle_correct
        if should_augment(augment):
            image = flip_image(image)
            angle_right = -angle_right

        images.append(image)
        angles.append(angle_right)

    return images, angles

