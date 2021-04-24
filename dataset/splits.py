import os
import numpy as np
import pandas as pd
# This class generates the splits and create links to the folder without moving the original files or
# create a copy of the current data.
class DataManager:
    def __init__(self, main_folder="dataset ML candidates/", test_split=0.5, seed=0):
        self.main_folder = main_folder
        self.test_split = test_split
        np.random.seed(seed)
    # Data must be split in validation and training.
    def Generate_Splits(self):
        # Data is not separated by train and test, it is separated by classes 
        all_cls = os.listdir(self.main_folder)
        all_cls = [a for a in all_cls if os.path.isdir(self.main_folder+a)]
        all_images = []
        all_labels = []
        for cls in all_cls:
            path = os.path.join(self.main_folder, cls)
            # "../" was added to solve quickly the path problems
            set_imgs = [os.path.join(path,img) for img in os.listdir(path) if img.endswith("jpg")]
            all_images.extend(set_imgs)
            all_labels.extend([cls]*len(set_imgs))
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        idx = np.arange(len(all_images))
        np.random.shuffle(idx)
        all_images = all_images[idx]
        all_labels = all_labels[idx]
        self.test_split = int(self.test_split*len(all_images))
        self.tr_data = all_images[self.test_split:]
        self.tr_label = all_labels[self.test_split:]

        self.va_data = all_images[:self.test_split]
        self.va_label = all_labels[:self.test_split]
    def Save_Splits(self):
        # Pandas was included for a more efficient link data storage
        data_train = pd.DataFrame({"imgs":self.tr_data, "labels":self.tr_label})
        data_validation = pd.DataFrame({"imgs":self.va_data, "labels":self.va_label})
        data_train.to_csv(self.main_folder+"training.csv")
        data_validation.to_csv(self.main_folder+"validation.csv")

    def Get_Splits(self):
        return self.tr_data, self.tr_label, self.va_data, self.va_label

if __name__ == '__main__':
    Manager = DataManager()
    Manager.Generate_Splits()
    Manager.Save_Splits()
