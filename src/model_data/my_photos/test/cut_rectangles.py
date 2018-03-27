import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# constants
image_folder = '.'

def get_rectangle(height, width, channels, box1, image):
    return image[box1["upper_left"][1] : box1["down_right"][1], box1["upper_left"][0] : box1["down_right"][0], :]


if __name__ == '__main__':
    
    for n, image_file in enumerate(os.scandir(image_folder)):
        path = image_file.path
        filename, file_extension = os.path.splitext(path)
        print(filename)
        if file_extension == ".png":
            json_file_name = filename + ".json"
            print(json_file_name)
            json_data = json.load(open(json_file_name))
            image = cv2.imread(image_file.path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            height = json_data["height"]
            width = json_data["width"]
            channels = json_data["channels"]
    
            i = 0
            for box in json_data["object"]:
                i=i+1
                rectangle = get_rectangle(height, width, channels, box, image)
                rectangle_file_name = filename + "_" + str(i) + ".png"
                rectangle = cv2.resize(rectangle, (200,200))
                cv2.imwrite(rectangle_file_name, rectangle)
                with open("labels.txt", 'a') as file:
                    file.write(rectangle_file_name + "\t")
                    file.write(box["piece"] + "\t")
                    file.write(box["piece_colour"]+ "\t")
                    file.write(box["field_colour"] + "\n")
    