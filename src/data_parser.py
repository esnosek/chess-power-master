import pandas as pd
import cv2 
import numpy as np

data_dir = "data/"
data_file = data_dir + "/labels.csv"
width = 200
height = 200
class_number = 2

def get_binary_labeled_data(gray = False):
    channels = 1 if gray else 3
    data = pd.read_csv(data_file, sep='\t')
    size = data["file_name"].size
    images = np.zeros((size, width, height, channels), dtype = 'float32')
    labels = np.zeros((size, class_number), dtype = 'float32')
    for i in range(data["file_name"].size):
        file_name = data_dir + data["file_name"][i]
        images[i] = cv2.imread(file_name, channels)
        if data["piece"][i] == "EMPTY":
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    
    return images.reshape((size, width*height*channels)), labels
    
    
if __name__ == '__main__':
    data_file = data_dir + "/labels.csv"
    images, labels = get_binary_labeled_data()