import os
import matplotlib.pyplot as plt
import argparse
# Usage
# python test_label.py --test1 test1 --test2 test2
ap = argparse.ArgumentParser()
ap.add_argument("-t1", "--test1", type=str, default='test1', help="path to no label test")
ap.add_argument("-t2", "--test2", type=str, default='test2', help="path to label test")
args = vars(ap.parse_args())
a = 'dog'
b = 'cat'
filenames = os.listdir(args['test1'])
f2 = os.listdir(args['test2'])
for filename in filenames:
    new_filename = filename.split('.')[0]
    i = filenames.index(filename)+1
    try:
        p = f2[i - 1].split('.')[0]
    except:
        print("正在标记第{}张照片,{}".format(i, filename))
        image = plt.imread(args['test1'] + '/' + filename)
        plt.imshow(image)
        plt.ion()
        plt.pause(1)
        plt.close('all')
        choice = input("请输入图片的类别，1 表示狗，0表示猫: ")
        if eval(choice) == 1:
            new_filename = new_filename + '.' + a + '.jpg'
        if eval(choice) == 0:
            new_filename = new_filename + '.' + b + '.jpg'
        plt.imsave(args['test2'] + '/' + new_filename, image)


    else:
        if new_filename != filename.split('.')[0]:
            print("正在标记第{}张照片,{}".format(i, filename))
            image = plt.imread(args['test1'] + '/' + filename)
            plt.imshow(image)
            plt.ion()
            plt.pause(1)
            plt.close('all')
            choice = input("请输入图片的类别，1 表示狗，0表示猫: ")
            if eval(choice) == 1:
                new_filename = new_filename + '.' + a + '.jpg'
            if eval(choice) == 0:
                new_filename = new_filename + '.' + b + '.jpg'
            plt.imsave(args['test2'] + '/' + new_filename, image)


