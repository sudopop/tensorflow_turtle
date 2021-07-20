# TF 데이터셋 변환 참고 : https://stackoverflow.com/questions/47182843/how-to-feed-feed-dict-in-tensorflow-my-own-image-data

# 참고 깃헙 : https://github.com/deep-diver/AlexNet/blob/master/alexnet.py

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

        # self.MaxIter = int(self.NumFiles / self.BatchSize)
        self.MaxIter = math.ceil(self.NumFiles / self.BatchSize)

        self.SuffleBatch()  # suffle file list

    def SuffleBatch(self):
        self.SFiles = []
        Sf = np.array(range(np.int32(np.ceil(self.NumFiles / self.BatchSize) + 1))) * self.BatchSize
        random.shuffle(Sf)

        for i in range(len(Sf)):
            for k in range(self.BatchSize):
                if Sf[i] + k < self.NumFiles:
                    self.SFiles.append(self.OrderedFiles[Sf[i] + k])

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

    def TrainBatch(self, epoch):
        if self.itr >= self.NumFiles:  # End of an epoch
            self.itr = 0
        batch_size = np.min([self.BatchSize, self.NumFiles - self.itr])
        labels = []
        images = []

        tmp_path = "./tmp_png"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
            os.makedirs(tmp_path)
        else:
            os.makedirs(tmp_path)

        # tmp_path2 = "./tmp_png2"
        # if os.path.exists(tmp_path2):
        #     shutil.rmtree(tmp_path2, ignore_errors=True)
        #     os.makedirs(tmp_path2)
        # else:
        #     os.makedirs(tmp_path2)

        for f in range(batch_size):

            img_name = self.Image_Dir + "/" + self.SFiles[self.itr]
            img = self.load_image(img_name)
            img = self.resize_image(img, 224, 224)
            tmp_img_name = tmp_path + "/" + self.SFiles[self.itr]
            img.save(tmp_img_name)


            print("abnormal!")
            size = min(img.size[0], img.size[1])
            # r_size = random.randint(5, size)  # 랜덤 사이즈

            x = random.randint(-50, 50)  # 이동
            y = random.randint(-50, 50)  # 랜덤하게 별의 위치

            turtle.speed(0)
            turtle.penup()

            turtle.setup(img.size[0], img.size[1])
            turtle.bgpic(tmp_img_name)  # 이름이 바뀌어야만 업데이트 됨! (동일이름은 이전에 작업한거랑 똑같다고 생각하고 업데이트 안됨!)
            turtle.colormode(255)  # random 컬러 해줄려면 먼저 주는 옵션
            turtle.goto(x, y)

            # turtle.begin_fill()  # 무조건 도형을 채우는 걸로

            for i in range(random.randint(1,3)):
                r_size = random.randint(5, 20)  # 랜덤 사이즈
                s = random.randint(2, 5)  # 도형의 형태 랜
                r = random.randint(100, 150)  # 색상
                g = random.randint(100, 150)  # 색상
                b = random.randint(100, 150)  # 색상
                p = random.randint(1, 10)  # 폰트 크가

                # turtle.bgpic(img_name)
                turtle.color(r, g, b)
                turtle.pensize(p)

                # if random.randint(0, 1): #도형을 채울지 말지 랜덤으로 결정
                #     turtle.begin_fill()

                turtle.pendown()

                for j in range(s):
                    turtle.forward(r_size)
                    turtle.left(360 / s)
                turtle.goto(x+(r_size*(i+1)), y+(r_size*(i+1)))
                turtle.penup()
                # turtle.end_fill()
                turtle.ht()

            turtle.getcanvas().postscript(file="2_temp.eps")
            # turtle.reset()

            pic = self.load_image("2_temp.eps")
            pic.save("2_temp.png")
            pic = turtle.getscreen().getcanvas().postscript(colormode="color")
            pic = Image.open(io.BytesIO(pic.encode("utf-8")))
            name11 = "./Data/create/" +str(epoch) +"_" + self.SFiles[self.itr]
            pic.save(name11, format="PNG")
            turtle.reset()

            img = self.resize_image(pic, 224, 224)  # resize를 위해서 해주니 주석처리!
            np_img = self.pil_to_nparray(img)  # pic 으로 해줄려고

            # tmp_img_name2 = tmp_path2 + "/" + self.SFiles[self.itr]
            # img.save(tmp_img_name2)

            # np_img = self.pil_to_nparray(pic) # 에러 발생 (eps로 바뀌면서 크기가 자동으로 작아짐.. 왜그런지는 모름ㅠ)

            pic.save("2_temp.png")
            images.append(np_img)
            label = np.zeros(self.NumClass)
            index = 1
            label[index] = 1
            labels.append(label)

            self.itr += 1

        return images, labels


class AlexNet:
    def __init__(self, num_classes, learning_rate):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self.global_step = tf.Variable(0, trainable=False, name='global_step') # optimizer 1번할때마다 1번씩 내부적으로 count

        self.logits = self.load_model()
        self.model = tf.identity(self.logits, name='logits')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label),
                                   name='cost')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam').minimize(self.cost, self.global_step)

        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

    def load_model(self):
        # 1st
        conv1 = conv2d(self.input, num_outputs=96,
                       kernel_size=[11, 11], stride=4, padding="VALID",
                       activation_fn=tf.nn.relu)
        lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001, beta=0.75)
        pool1 = max_pool2d(lrn1, kernel_size=[3, 3], stride=2)
        # pool1 = max_pool2d(conv1, kernel_size=[3, 3], stride=2)

        # 2nd
        conv2 = conv2d(pool1, num_outputs=256,
                       kernel_size=[5, 5], stride=1, padding="VALID",
                       biases_initializer=tf.ones_initializer(),
                       activation_fn=tf.nn.relu)
        lrn2 = tf.nn.local_response_normalization(conv2, bias=2, alpha=0.0001, beta=0.75)
        pool2 = max_pool2d(lrn2, kernel_size=[3, 3], stride=2)
        # pool2 = max_pool2d(conv2, kernel_size=[3, 3], stride=2)

        # 3rd
        conv3 = conv2d(pool2, num_outputs=384,
                       kernel_size=[3, 3], stride=1, padding="VALID",
                       activation_fn=tf.nn.relu)

        # 4th
        conv4 = conv2d(conv3, num_outputs=384,
                       kernel_size=[3, 3], stride=1, padding="VALID",
                       biases_initializer=tf.ones_initializer(),
                       activation_fn=tf.nn.relu)

        # 5th
        conv5 = conv2d(conv4, num_outputs=256,
                       kernel_size=[3, 3], stride=1, padding="VALID",
                       biases_initializer=tf.ones_initializer(),
                       activation_fn=tf.nn.relu)
        pool5 = max_pool2d(conv5, kernel_size=[3, 3], stride=2)

        # 6th
        flat = flatten(pool5)
        fcl1 = fully_connected(flat, num_outputs=4096,
                               biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        dr1 = tf.nn.dropout(fcl1, 0.5)

        # 7th
        fcl2 = fully_connected(dr1, num_outputs=4096,
                               biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        dr2 = tf.nn.dropout(fcl2, 0.5)

        # output
        out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)
        return out

    def train(self, Train_Image_Dir, Validate_Image_Dir, Epochs, Batch_Size, save_model_path, save_num = 10):
        TrainReader = Data_Reader(ImageDir=Train_Image_Dir, BatchSize=Batch_Size, NumClass=self.num_classes)

        for epoch in range (2):
            for iter in range(TrainReader.MaxIter):
                Images, GTLabels = TrainReader.TrainBatch(epoch)



def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running AlexNet')
    parser.add_argument('--train-dataset-path', help='location where the dataset is present', default='./Data/train')
    parser.add_argument('--validate-dataset-path', help='location where the dataset is present', default='./Data/test')
    parser.add_argument('--save-path', default='./Log_1203_6/')
    parser.add_argument('--learning-rate', help='learning rate', default=0.001)
    parser.add_argument('--epochs', default=301)
    parser.add_argument('--batch-size', default=70)
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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    alexNet = AlexNet(class_num, learning_rate)
    alexNet.train(trian_dataset_path, validate_dataset_path, epochs, batch_size, save_path)


if __name__ == "__main__":
    main()
