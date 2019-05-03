import pandas as pd
import cv2 
import numpy as np

data_dir = "../data/"
data_file = data_dir + "train_data/labels.csv"
width = 50
height = 50
channels = 3
class_number = 2


def get_binary_labeled_data(test, one_hot = False):
    
    data = pd.read_csv(data_file, sep='\t')
    data = data.sample(frac = 1).reset_index(drop=True)
    size = data["file_name"].size
    images = np.zeros((size, width, height, channels), dtype = 'float32')
    labels = np.zeros((size), dtype = 'float32') 
    one_hot_labels = np.zeros((size, class_number), dtype = 'float32')
    
    for i in range(size):
        file_name = data_dir + data["file_name"][i]
        img = cv2.imread(file_name, channels)
        images[i] = img
        class_index = 0 if data["piece"][i] == "EMPTY" else 1
        one_hot_labels[i][class_index] = 1
        labels[i] = 0 if data["piece"][i] == "EMPTY" else 1
    
    end = int(test * size)
    
    train_images = images[0:end, :]
    train_labels = labels[0:end]
    train_one_hot_labels = one_hot_labels[0:end, :]

    test_images = images[end:, :]
    test_labels = labels[end:]
    test_one_hot_labels = one_hot_labels[end:, :]
    
    if one_hot: 
        return train_images, train_one_hot_labels, test_images, test_one_hot_labels
    else:
        return train_images, train_labels, test_images, test_labels
    

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = get_binary_labeled_data(0.8)
