import cv2
import json
import os
from os.path import basename

# constants
json_folder = '.'

def get_rectangle(height, width, channels, box1, image):
    return image[box1["upper_left"][1] : box1["down_right"][1], box1["upper_left"][0] : box1["down_right"][0], :]


if __name__ == '__main__':
    for n, file in enumerate(os.scandir(json_folder)):
        path = file.path
        filename, file_extension = os.path.splitext(path)
        if file_extension == ".json":
            img_file_name = filename + ".png"
            print(basename(filename))
            json_data = json.load(open(path))
            image = cv2.imread(img_file_name)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            height = json_data["height"]
            width = json_data["width"]
            channels = json_data["channels"]
    
            i = 0
            for box in json_data["object"]:
                i=i+1
                rectangle = get_rectangle(height, width, channels, box, image)
                rectangle_file_name = "fields/" + basename(filename) + "_" + str(i) + ".png"
                rectangle = cv2.resize(rectangle, (200,200))
                cv2.imwrite(rectangle_file_name, rectangle)
                with open("labels.csv", 'a') as file:
                    file.write(rectangle_file_name + "\t")
                    file.write(box["piece"] + "\t")
                    file.write(box["piece_colour"]+ "\t")
                    file.write(box["field_colour"] + "\n")
    