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
    # 4 * size because of image rotation
    images = np.zeros((4 * size, width, height, channels), dtype = 'float32')
    labels = np.zeros((4 * size), dtype = 'float32') 
    one_hot_labels = np.zeros((4 * size, class_number), dtype = 'float32')
    
    #rotation matrixes
    M90 = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1.0)
    M180 = cv2.getRotationMatrix2D((width / 2, height / 2), 180, 1.0)
    M270 = cv2.getRotationMatrix2D((width / 2, height / 2), 270, 1.0)
    
    for i in range(size):
        file_name = data_dir + data["file_name"][i]
        img = cv2.imread(file_name, channels)
        images[i] = img
        images[size + i] = cv2.warpAffine(img, M90, (width, height))
        images[2 * size + i] = cv2.warpAffine(img, M180, (width, height))
        images[3 * size + i] = cv2.warpAffine(img, M270, (width, height))
        class_index = 0 if data["piece"][i] == "EMPTY" else 1
        one_hot_labels[i][class_index] = 1
        one_hot_labels[size + i][class_index] = 1
        one_hot_labels[2 * size + i][class_index] = 1
        one_hot_labels[3 * size + i][class_index] = 1
        label = 0 if data["piece"][i] == "EMPTY" else 1
        labels[i] = label
        labels[size + i] = label
        labels[2 * size + i] = label
        labels[3 * size + i] = label
    
    end = int(test * size * 4)
    
    
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
