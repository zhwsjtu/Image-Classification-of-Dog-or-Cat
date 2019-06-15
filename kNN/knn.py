import numpy as np
import cv2
import os
import math
import time

start = time.perf_counter()
RATIO = 0.2
train_dir = 'D:/Projects/dogorcat_kNN/dataset/train/'
#modify the directory of train dataset.
#first use file folder /train2(1000 cats and 1000 dogs)to choose k
#then use file folder /train(12500 cats and 12500 dogs)to test

# the directory and labels of images in training set and  test set are acquired according to the parameter RATIO.
def get_files(file_dir, ratio):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('数据集中有 %d cats\n数据集中有 %d dogs' % (len(cats), len(dogs)))
    # 图片list和标签list
    # hstack 水平(按列顺序)把数组给堆叠起来
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    # 根据比率，确定训练和测试数量
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples
    tra_images = []
    val_images = []
    # 按照0-n_train为tra_images，后面位val_images的方式来排序

    for index in range(n_train):
        image = cv2.imread(all_image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (32, 32))
        tra_images.append(image)
    tra_labels = all_label_list[:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]

    for index in range(n_val):
        image = cv2.imread(all_image_list[n_train + index])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (32, 32))
        val_images.append(image)
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


if __name__ == '__main__':
    _train, train_labels, _val, val_labels = get_files(train_dir, RATIO)

    x = np.array(_train)
    train = x.reshape(-1, 32 * 32).astype(np.float32)
    y = np.array(_val)
    test = y.reshape(-1, 32 * 32).astype(np.float32)
    # create kNN model
    knn = cv2.ml.KNearest_create()
    knn.train(np.array(train), cv2.ml.ROW_SAMPLE, np.array(train_labels))
    ret, result, neighbours, dist = knn.findNearest(test, k=10)
    # check the accurracy of classification through comparing the trained label with correct label
    np_val_labels = np.array(val_labels)[:, np.newaxis]
    matches = result == np_val_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print("the accuracy is %f percent" % (accuracy))
    end = time.perf_counter()
    print('Running time: %f Seconds' % (end - start))


