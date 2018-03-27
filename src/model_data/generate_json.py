import json
import cv2
import os

def write_json(img, boxes):
    filename, file_extension = os.path.splitext(img)
    img = cv2.imread(img.path)
    height, width, channels = img.shape
    json_file = json.dumps({"height" : height, "width" : width, "channels" : channels, "object" : [box.__dict__ for box in boxes]})
    
    with open(filename + '.json', 'w') as file:
        file.write(json_file)
    