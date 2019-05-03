import cv2
import json
import os
from os.path import basename

# constants
json_folder = 'tagged_chessboard_fields'
width = 50
height = 50
labels_filename = "train_data/labels.csv"

def get_rectangle(box, image):
    return image[box["upper_left"][1] : box["down_right"][1], box["upper_left"][0] : box["down_right"][0], :]

def save_fields_as_image():
    # create labels.csv file with headers
    with open(labels_filename, 'a') as file:
        file.write("file_name" + "\t")
        file.write("piece" + "\t")
        file.write("piece_colour"+ "\t")
        file.write("field_colour" + "\n")
    
    for n, file in enumerate(os.scandir(json_folder)):
        filename, file_extension = os.path.splitext(file.path)
        if file_extension == ".json":
            print(basename(filename))
            
            json_data = json.load(open(file.path))
            img_file_name = filename + ".png"
            image = cv2.imread(img_file_name)
    
            i = 0
            for box in json_data["object"]:
                i=i+1
                dir_name = "train_data/" + box["piece"]
                if not os.path.isdir(dir_name): os.mkdir(dir_name)
                rectangle_file_name = dir_name + "/" + basename(filename) + "_" + str(i) + ".png"
                print(rectangle_file_name)
                
                rectangle = cv2.resize(get_rectangle(box, image), (width, height))
                cv2.imwrite(rectangle_file_name, rectangle)
                with open(labels_filename, 'a') as file:
                    file.write(rectangle_file_name + "\t")
                    file.write(box["piece"] + "\t")
                    file.write(box["piece_colour"]+ "\t")
                    file.write(box["field_colour"] + "\n")
                    
if __name__ == '__main__':
    save_fields_as_image()
    