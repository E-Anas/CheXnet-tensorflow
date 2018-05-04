import tensorflow as tf
import PIL
from PIL import Image
import os
import numpy as np




class ChestXrayDataSet():
    def __init__(self, data_dir, image_list_file,IMG_SIZE):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.IMG_SIZE=IMG_SIZE

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        image = image.resize([self.IMG_SIZE,self.IMG_SIZE], resample=PIL.Image.BILINEAR)
        label = self.labels[index]
        return image , label

    def __len__(self):
        return len(self.image_names)