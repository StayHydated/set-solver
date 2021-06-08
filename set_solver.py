import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import math
from object_detection.utils import label_map_util
from PIL import Image

# card attributes
RED = 0
GREEN = 1
PURPLE = 2
SOLID = 0
EMPTY = 1
STRIPED = 2

# open the tensoflow model
model = tf.saved_model.load('saved_model')
category_index = label_map_util.create_category_index_from_labelmap('saved_model/label_map.pbtxt', use_display_name=True)

def detect_objects(filename):
    # load image into numpy array
    img = np.array(Image.open(filename))
    # convert image to tensor
    input_tensor = tf.convert_to_tensor(img)
    # the model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    detections = model_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    return detections


class Card:
    def __init__(self, relative_points, img_height, img_width):
        self.y1 = round(relative_points[0]*img_height)
        self.x1 = round(relative_points[1]*img_width)
        self.y2 = round(relative_points[2]*img_height)
        self.x2 = round(relative_points[3]*img_width)
        self.center = ((self.x1+self.x2)/2, (self.y1+self.y2)/2)
        self.shapes = []

class Shape:
    def __init__(self, relative_points, img_height, img_width):
        self.y1 = round(relative_points[0]*img_height)
        self.x1 = round(relative_points[1]*img_width)
        self.y2 = round(relative_points[2]*img_height)
        self.x2 = round(relative_points[3]*img_width)
        self.center = ((self.x1+self.x2)/2, (self.y1+self.y2)/2)

# iterate through the detected objects and create a list of deteccted cards
def create_cards(detections, img_height, img_width):
    cards = []
    detection_threshold = 0.9
    card_class = 4
    for idx, detection_score in enumerate(detections['detection_scores']):
        if (detection_score > detection_threshold and detections['detection_classes'][idx] == card_class):
            card = Card(detections['detection_boxes'][idx], img_height, img_width)
            cards.append(card)
    return cards

# iterate through detected objects and assign shapes to appropriate cards
def add_shapes_to_cards(cards, detections, img_height, img_width):
    detection_threshold = 0.9
    card_class = 4
    for idx, detection_score in enumerate(detections['detection_scores']):
        if (detection_score > detection_threshold and detections['detection_classes'][idx] != card_class):
            shape = Shape(detections['detection_boxes'][idx], img_height, img_width)
            shape.shape = detections['detection_classes'][idx]
            # add the newly created shape to the card with the lowest distance from it
            lowest_dist = max(img_height, img_width)
            for card in cards:
                dist = math.dist(card.center, shape.center)
                if (dist < lowest_dist):
                    lowest_dist = dist
                    closest_card = card
            closest_card.shapes.append(shape)

# determine shape, number, color, shading of cards
def determine_attributes(cards, imgcv):
    imghsv = cv.cvtColor(imgcv, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(imgcv, cv.COLOR_BGR2GRAY)
    # determine if image is set in low-light conditions
    mean, std = cv.meanStdDev(img_gray)
    if (mean < 125):
        lowlight = True
    else:
        lowlight = False
    red_lo_H, red_lo_S, red_lo_V = 0, 40, 0
    red_hi_H, red_hi_S, red_hi_V = 30, 255, 255
    ##
    green_lo_H, green_lo_S, green_lo_V = 40, 40, 0
    green_hi_H, green_hi_S, green_hi_V = 80, 255, 255
    ##
    purple_lo_H, purple_lo_S, purple_lo_V = 110, 10, 0
    purple_hi_H, purple_hi_S, purple_hi_V = 175, 255, 255
    for card in cards:
        card.number = len(card.shapes) - 1
        # the number of red, green, and purple pixels
        threshold_r, threshold_g, threshold_p = 0, 0, 0
        shape_possibilities = []
        grayscale_mean, grayscale_std = 0, 0
        for shape in card.shapes:
            shape_possibilities.append(shape.shape)
            # add the number of red, green, and purple pixels of each shape has to the threshold sum
            threshold_r += cv.inRange(imghsv[shape.y1:shape.y2, shape.x1:shape.x2], (red_lo_H, red_lo_S, red_lo_V), (red_hi_H, red_hi_S, red_hi_V)).sum()
            threshold_g += cv.inRange(imghsv[shape.y1:shape.y2, shape.x1:shape.x2], (green_lo_H, green_lo_S, green_lo_V), (green_hi_H, green_hi_S, green_hi_V)).sum()
            threshold_p += cv.inRange(imghsv[shape.y1:shape.y2, shape.x1:shape.x2], (purple_lo_H, purple_lo_S, purple_lo_V), (purple_hi_H, purple_hi_S, purple_hi_V)).sum()
            # take a sample from the center of a shape using 10% of the width and height to determine shading
            rect_width = shape.x2 - shape.x1
            rect_height = shape.y2 - shape.y1
            sample_y1 = shape.y1 + round(rect_height*0.45)
            sample_y2 = shape.y2 - round(rect_height*0.45)
            sample_x1 = shape.x1 + round(rect_width*0.45)
            sample_x2 = shape.x2 - round(rect_width*0.45)
            sample_mean, sample_std = cv.meanStdDev(img_gray[sample_y1:sample_y2, sample_x1:sample_x2])
            grayscale_mean += sample_mean
            grayscale_std += sample_std
        if (len(shape_possibilities) < 1 or len(shape_possibilities) > 3):
            cards.remove(card)
            continue
        # the card's shape is the most common shape in its card list
        card.shape = max(set(shape_possibilities), key = shape_possibilities.count) - 1
        # the card's shape is the threshold with the most pixels
        threshold_max = max(threshold_r, threshold_g, threshold_p)
        if (threshold_max == threshold_r):
            card.color = RED
        elif (threshold_max == threshold_g):
            card.color = GREEN
        elif (threshold_max == threshold_p):
            card.color = PURPLE
        # the card's shading is a combination of the sample's grayscale mean and standard deviation
        grayscale_mean /= len(card.shapes)
        grayscale_std /= len(card.shapes)
        if lowlight:
            if (grayscale_mean < 70):
                card.shading = SOLID
            elif (grayscale_std < 4.5):
                card.shading = EMPTY
            else:
                card.shading = STRIPED
        else:
            if (grayscale_mean < 110):
                card.shading = SOLID
            elif (grayscale_std < 10):
                card.shading = EMPTY
            else:
                card.shading = STRIPED

# find sets and label them
def find_sets(cards, imgcv):
    pixel_offset = 0
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            for k in range(j + 1, len(cards)):
                number_modulus = (cards[i].number + cards[j].number + cards[k].number)%3
                shape_modulus = (cards[i].shape + cards[j].shape + cards[k].shape)%3
                color_modulus = (cards[i].color + cards[j].color + cards[k].color)%3
                shading_modulus = (cards[i].shading + cards[j].shading + cards[k].shading)%3
                if (number_modulus == shape_modulus == color_modulus == shading_modulus ==0):
                    color = list(np.random.random(size=3) * 256)
                    cv.rectangle(imgcv, (cards[i].x1 + pixel_offset, cards[i].y1 + pixel_offset), (cards[i].x2 - pixel_offset, cards[i].y2 - pixel_offset), color, 3)
                    cv.rectangle(imgcv, (cards[j].x1 + pixel_offset, cards[j].y1 + pixel_offset), (cards[j].x2 - pixel_offset, cards[j].y2 - pixel_offset), color, 3)
                    cv.rectangle(imgcv, (cards[k].x1 + pixel_offset, cards[k].y1 + pixel_offset), (cards[k].x2 - pixel_offset, cards[k].y2 - pixel_offset), color, 3)
                    pixel_offset += 5


def main():
    images = []
    images_dir = 'images_to_evaluate/'
    for file in os.listdir(images_dir):
        # using tensorflow object detection, create a dictionary of detected cards and shapes
        detections = detect_objects(images_dir+file)
        imgcv = cv.imread(images_dir+file, cv.IMREAD_COLOR)
        # create a list of card objects
        img_height = imgcv.shape[0]
        img_width = imgcv.shape[1]
        cards = create_cards(detections, img_height, img_width)
        # add detected shapes to the appropriate cards
        add_shapes_to_cards(cards, detections, img_height, img_width)
        # determine shape, number, color, shading of cards
        determine_attributes(cards, imgcv)
        # find and label sets
        find_sets(cards, imgcv)
        images.append((file, imgcv))

    for idx, image in enumerate(images):
        cv.imshow(image[0], image[1])

    cv.waitKey()

if __name__ == "__main__":
    main()













###
