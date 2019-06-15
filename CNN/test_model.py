import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import time
# Usage
# select the model location, and then modify the path value
# 模型位置
path = r'E:\Users\python\ml_project\model\cat_dog.model'
model = load_model(filepath=path)

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 64
test_path = 'test2'

# 读取测试图片


def get_data():
    categories = []
    test_filenames = os.listdir(test_path)
    for test_filename in test_filenames:
        category = test_filename.split('.')[1]
        if category == 'dog':
            categories.append(1)  # 狗的标签是1
        else:
            categories.append(0)
    df = pd.DataFrame({'filename': test_filenames, 'category': categories})
    df['category'] = df['category'].astype('int')
    return df

def predict(model=model):
    test_df = get_data()
    nb_samples = test_df.shape[0]
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        test_path,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )
    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))
    threshold = 0.5
    test_df['probability'] = predict
    test_df['predict'] = np.where(test_df['probability'] > threshold, 1, 0)  # 模型预测
    right_count = 0
    for index, row in test_df.iterrows():
        category = row['category']
        predict = row['predict']
        if category == predict:
            right_count = right_count + 1
    acc = right_count/test_df.shape[0]
    return test_df, acc
def check_result(test_df):
    '''查看部分结果'''
    num = 12
    if num < test_df.shape[0]:
        sample_test = test_df.sample(n=num)
    else:
        sample_test = test_df
    i = 0
    plt.figure(figsize=(12, 12))
    for index, row in sample_test.iterrows():
        i = i + 1
        filename = row['filename']
        category = row['predict']
        if category == 1:
            label = 'Dog'
        else:
            label = 'Cat'
        probability = row['probability']
        img = load_img(test_path + '/' + filename, target_size=IMAGE_SIZE)
        plt.subplot(4, num //4, i)
        plt.imshow(img)
        plt.title(filename + '  ' + "{}".format(label) + '/' + "{}".format(round(probability, 2)))

    plt.tight_layout()
    plt.savefig('test.png')
    plt.show()
if __name__ == '__main__':
    start = time.perf_counter()
    test_df, acc = predict()
    end = time.perf_counter()
    print("test time on {} samples is {:.2f} s".format(test_df.shape[0], end - start))
    print('在测试集上准确率 {:.2f} %'.format(acc*100))
    check_result(test_df)
