import os
import numpy as np

class DataManager:
    def __init__(self, main_folder="dataset ML candidates/", test_split=0.5, seed=0):
        self.main_folder = main_folder
        self.test_split = test_split
        np.random.seed(seed)
    # Data must be split in validation and training.
    def Generate_Splits(self):
        # Data is not separated by train and test, it is separated by classes 
        all_cls = os.listdir(self.main_folder)
        all_images = []
        all_labels = []
        for cls in all_cls:
            path = self.main_folder + cls
            set_imgs = [path+img for img in os.listdir(path) if img.endswith("jpg")]
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
        with open(self.main_folder+'/training_images.txt', 'w') as f:
            for item in self.tr_data: f.write("%s\n" % item)
        with open(self.main_folder+'/training_labels.txt', 'w') as f:
            for item in self.tr_label: f.write("%s\n" % item)


        with open(self.main_folder+'/validation_images.txt', 'w') as f:
            for item in self.va_data: f.write("%s\n" % item)
        with open(self.main_folder+'/validation_labels.txt', 'w') as f:
            for item in self.va_label: f.write("%s\n" % item)
    def Get_Splits(self):
        return self.tr_data, self.tr_label, self.va_data, self.va_label

if __name__ == '__main__':
    Manager = DataManager()
    Manager.Generate_Splits()
    Manager.Save_Splits()
