import cv2
import json
import os
import numpy as np
from os.path import basename

# constants
json_folder = 'tagged_chessboard_fields'
width = 50
height = 50
base_dir = "train_data/pieces_50x50"
test_to_validation_size = 0.9
labels_filename = base_dir + "/labels.csv"

def save_fields_as_image(create_csv=False):
    
    if not os.path.isdir(base_dir): 
        os.mkdir(base_dir)
    if not os.path.isdir(base_dir + "/" + "train"): 
        os.mkdir(base_dir + "/" + "train")
    if not os.path.isdir(base_dir + "/" + "validation"): 
        os.mkdir(base_dir + "/" + "validation")
        
    if create_csv:
        with open(labels_filename, 'a') as file:
            file.write("file_name" + "\t")
            file.write("piece" + "\t")
            file.write("piece_colour"+ "\t")
            file.write("field_colour" + "\n")
    
    images_by_class = {}
    for n, file in enumerate(os.scandir(json_folder)):
        filename, file_extension = os.path.splitext(file.path)
        if file_extension == ".json":
            json_data = json.load(open(file.path))
            img_file_name = filename + ".png"
            image = cv2.imread(img_file_name)        

            for idx, box in enumerate(json_data["object"]):
                # class_name = box["piece"] if box["piece"] == "EMPTY" else "OCCUPIED"
                class_name = box["piece"] 
                # class_name = box["field_colour"]
                
                if class_name not in images_by_class:
                    images_by_class[class_name] = []
                
                rectangle = cv2.resize(get_rectangle(box, image), (width, height))
                images_by_class[class_name].append(rectangle)
    
    print("Class names: " + str(images_by_class.keys()))
                    
    for class_name in images_by_class.keys():
        
        class_size = len(images_by_class[class_name])
        test_size = int(test_to_validation_size * class_size)
        validation_size = class_size - test_size
        
        print("Class " + class_name + ":\n\tsize: " + str(class_size) + 
              "\n\ttest size:" + str(test_size) + 
              "\n\tvalidation size: " + str(validation_size))
        
        for idx, rectangle in enumerate(images_by_class[class_name]):
            
            train_or_validation = "train" if idx < test_size else "validation" 
            dir_path = base_dir + "/" + train_or_validation + "/" + class_name
            if not os.path.isdir(dir_path): 
                os.mkdir(dir_path)
                
            rectangle_file_name = dir_path + "/" + class_name + "_" + str(idx) + ".png"
            cv2.imwrite(rectangle_file_name, rectangle)
            if create_csv:
                with open(labels_filename, 'a') as file:
                    file.write(rectangle_file_name + "\t")
                    file.write(box["piece"] + "\t")
                    file.write(box["piece_colour"]+ "\t")
                    file.write(box["field_colour"] + "\n")

def get_rectangle(box, image):
    return image[box["upper_left"][1] : box["down_right"][1], box["upper_left"][0] : box["down_right"][0], :]

                    
if __name__ == '__main__':
    save_fields_as_image(create_csv=False)
    