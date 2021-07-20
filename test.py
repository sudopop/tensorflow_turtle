# TF 데이터셋 변환 참고 : https://stackoverflow.com/questions/47182843/how-to-feed-feed-dict-in-tensorflow-my-own-image-data

#참고 깃헙 : https://github.com/deep-diver/AlexNet/blob/master/alexnet.py

import argparse
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected
import os
from os.path import join
from PIL import Image
import random
import matplotlib.pyplot as plt
import turtle
import shutil
import io
import math


class Data_Reader:
    def __init__(self, ImageDir, GTLabelDir="", BatchSize=1, NumClass=2):
        self.Image_Dir = ImageDir
        self.itr = 0  # Iteration
        self.NumFiles = 0  # Number of files in reader
        self.BatchSize = BatchSize

        self.OrderedFiles = []
        self.OrderedFiles += [each for each in os.listdir(self.Image_Dir) if
                              each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith(
                                  '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith(
                                  '.tif') or each.endswith('.gif')]  # Get list of training images
        self.OrderedFiles.sort()
        self.NumFiles = len(self.OrderedFiles)
        self.NumClass = NumClass

        # self.MaxIter = int(self.NumFiles / self.BatchSize) + 1
        self.MaxIter = math.ceil(self.NumFiles / self.BatchSize)

        self.SuffleBatch()  # suffle file list

    def SuffleBatch(self):
        self.SFiles = []
        Sf=np.array(range(np.int32(np.ceil(self.NumFiles/self.BatchSize)+1)))*self.BatchSize
        random.shuffle(Sf)

        for i in range(len(Sf)):
            for k in range(self.BatchSize):
                  if Sf[i]+k<self.NumFiles:
                      self.SFiles.append(self.OrderedFiles[Sf[i]+k])

    def load_image(self, img_path):
        img = Image.open(img_path)
        return img

    def resize_image(self, in_image, new_width, new_height, out_image=None,
                     resize_mode=Image.ANTIALIAS):
        img = in_image.resize((new_width, new_height), resize_mode)
        if out_image:
            img.save(out_image)
        return img

    def pil_to_nparray(self, pil_image):
        pil_image.load()
        return np.asarray(pil_image, dtype="float32")

    def ValidateBatch(self):
        if self.itr >= self.NumFiles:  # End of an epoch
            self.itr = 0
        batch_size = np.min([self.BatchSize, self.NumFiles - self.itr])
        labels = []
        images = []

        for f in range(batch_size):
            img = self.load_image(self.Image_Dir + "/" + self.SFiles[self.itr])
            img = self.resize_image(img, 224, 224)
            np_img = self.pil_to_nparray(img)
            images.append(np_img)

            label = np.zeros(self.NumClass)
            index = int(self.SFiles[self.itr].split('_')[-1].split('.')[0])
            label[index] = 1
            labels.append(label)
            self.itr += 1

        return images, labels


def pred(Train_Image_Dir, Validate_Image_Dir, Epochs, Batch_Size, save_model_path):
    UseValidate = True
    ValideReader = Data_Reader(ImageDir=Validate_Image_Dir, BatchSize=Batch_Size, NumClass=2)
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

        print('global_variables_initializer...')
        sess = tf.Session(graph=loaded_graph)
        loader = tf.train.import_meta_graph(save_model_path + 'model.ckpt-100.meta')
        loader.restore(sess, tf.train.latest_checkpoint(save_model_path))
        loaded_x = loaded_graph.get_tensor_by_name('input:0')
        loaded_y = loaded_graph.get_tensor_by_name('label:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_accuracy = loaded_graph.get_tensor_by_name('accuracy:0')

        print('starting predict ... ')

        val_acc = 0

        for iter in range(ValideReader.MaxIter):
            Images, GTLabels = ValideReader.ValidateBatch()
            val_acc += sess.run(loaded_accuracy,
                                feed_dict={loaded_x: Images, loaded_y: GTLabels})
            # preds = sess.run(tf.argmax(loaded_logits, 1), feed_dict={loaded_x: Images})
            # print(preds)


        print(" Validate Acc=" + str(val_acc / ValideReader.MaxIter))


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running AlexNet')
    parser.add_argument('--train-dataset-path', help='location where the dataset is present', default='./Data/train')
    parser.add_argument('--validate-dataset-path', help='location where the dataset is present', default='./Data/test')
    parser.add_argument('--save-path', default='./Log_1203_3/')
    parser.add_argument('--learning-rate', help='learning rate', default=0.0001)
    parser.add_argument('--epochs', default=151)
    parser.add_argument('--batch-size', default=1)
    parser.add_argument('--class-num', default=2)

    return parser.parse_args(args)

def main():
    args = sys.argv[1:]
    args = parse_args(args)

    trian_dataset_path = args.train_dataset_path
    validate_dataset_path = args.validate_dataset_path
    save_path = args.save_path
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    class_num = args.class_num

    pred(trian_dataset_path, validate_dataset_path, epochs, batch_size, save_path)

if __name__ == "__main__":
    main()
