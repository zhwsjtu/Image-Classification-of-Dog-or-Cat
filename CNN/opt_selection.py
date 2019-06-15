import matplotlib.pyplot as plt
from train_model import T_M
import pandas as pd
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np

# Usage
# python opt_selection.py --model model/cat_dog1.model --plot plot1.png


def get_data():
    '''读取测试图片'''
    categories = []
    test_filenames = os.listdir("test2")
    for test_filename in test_filenames:
        category = test_filename.split('.')[1]
        if category == 'dog':
            categories.append(1)  # 狗的标签是1
        else:
            categories.append(0)
    df = pd.DataFrame({'filename': test_filenames, 'category': categories})
    df['category'] = df['category'].astype('int')
    return df


def predict(model,df):
    test_df = df
    nb_samples = test_df.shape[0]
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        "test2/",
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
    return acc

def draw_bar(opts,accs):
    width = 0.4
    ind = np.linspace(0.5,9.5,len(opts))
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.bar(ind-width/2,accs,width,color='green')
    ax.set_xticks(ind)
    ax.set_xticklabels(opts)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Accuracy on test dataset')
    ax.set_title('Different Optimizer influence', bbox={'facecolor':'0.8', 'pad':5})
    plt.grid(True)
    plt.savefig("Optimizer1.jpg")
    plt.show()
    plt.close()


opts = ['sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam']
accs = []

# 设置
IMAGE_WIDTH = T_M().IMAGE_WIDTH
IMAGE_HEIGHT = T_M().IMAGE_HEIGHT
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = T_M().batch_size
df = get_data()
for opt in opts:
    print("[INFO] 选择优化器{} 模型训练".format(opt))
    T_M(opt,1000).model_train()
    # 模型位置
    path = T_M().args['model']
    model = load_model(filepath=path)
    a = predict(model,df)
    accs.append(a)
    print('[INF0] 准备下一次训练')
    time.sleep(2)

draw_bar(opts, accs)



