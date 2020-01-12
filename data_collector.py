import cv2
import os
import json
import numpy as np


final_list = []

#"/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tdcb_labelData_train/labelData/train/tsinghuaDaimlerDataset/"
def load_labels(filename, json_folder_path="path/to/dataset/json"):
    annotation_data, meta_data, intermidiate_data = [],[],[]
    json_files = [ x for x in os.listdir(json_folder_path) if x == filename ]
    json_data = list()
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        with open(json_file_path, "rb") as f:
            json_data.append(json.load(f))
    for element in json_data:
        meta_data = []
        data = element['children'][0]
        meta_data.append([data['minrow']//4, data['mincol']//8, data['maxrow']//4, data['maxcol']//8])
    annotation_data.append(meta_data)

    return annotation_data[0]
    
    
def load_images_from_folder (folder='path/to/images'):
    images = []
    for filename in os.listdir(folder):
        try :
            if filename.endswith('png') :
                img = cv2.imread(os.path.join(folder,filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                filename_to_be_passed = filename[:-15] + 'labelData.json'
                labels = load_labels(filename_to_be_passed)
                images.append([img, labels[0]])
                print(filename_to_be_passed)
        except:
            pass

    return images

data = load_images_from_folder()
np.save('training_data1.npy', data)




