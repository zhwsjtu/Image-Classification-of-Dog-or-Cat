import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time

filenames = os.listdir("train")
class Data():
    def __init__(self,number=12500):
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 128
        self.IMAGE_SIZE = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.IMAGE_CHANNELS = 3  # RGB color
        self.filenames = os.listdir("train")
        self.batch_size = 64
        self.number = number  # 调整用于训练的每类图片数量

    def getdata(self):
        categories = []
        file_name = []
        number = self.number
        for filename in self.filenames:
            category = filename.split('.')[0]
            if category == 'dog':
                if eval(filename.split('.')[1]) <= number - 1:
                    categories.append(1)  # 狗的标签是1
                    file_name.append(filename)

            else:
                if eval(filename.split('.')[1]) <= number - 1:
                    categories.append(0)
                    file_name.append(filename)

        df = pd.DataFrame({'filename': file_name, 'category': categories})
        df['category'] = df['category'].astype('str')

        train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)  # 重新编号
        validate_df = validate_df.reset_index(drop=True)
        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]

        # 图片偏移，增加数据样本和训练可靠性
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            "train",
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='binary',
            batch_size=self.batch_size
        )

        validation_datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df,
            "train",
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='binary', # 该模式要声明category的类型是str
            batch_size=self.batch_size
        )
        return train_df, validate_df, train_generator, validation_generator, \
               total_train, total_validate

    def check_picture(self): # 查看几张照片

        samples = np.random.choice(self.filenames, size=10)
        plt.subplots(2, 5)
        for i in range(len(samples)):
            image = load_img("train/" + samples[i], target_size=self.IMAGE_SIZE)
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)
            plt.title(samples[i])
        plt.tight_layout()
        plt.show()

    def generator_work(self, get_data):
        train_df = get_data[0]
        # train_df['category'] = train_df['category'].astype('str')
        example_df = train_df.sample(n=1).reset_index(drop=True)
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        example_generator = train_datagen.flow_from_dataframe(
            example_df,
            "train",
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='input')
        plt.figure(figsize=(24, 24))
        for i in range(0, self.batch_size):
            plt.subplot(8, self.batch_size//9 + 1, i+1)
            for X_batch, Y_batch in example_generator:
                image = X_batch[0]
                plt.imshow(image)
                break
        plt.tight_layout()
        plt.savefig('sample.png')
        plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    print("it is working, please wait....")
    Data = Data(100)
    get_data = Data.getdata()
    Data.generator_work(get_data)
    end = time.perf_counter()
    print("运行时间：{:.3f} s".format(end - start))
