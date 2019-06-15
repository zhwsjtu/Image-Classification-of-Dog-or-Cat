from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from pre_data import Data
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
import argparse
import tensorflow as tf
import time

# Usage
# python train_model.py --model model/cat_dog2.model --plot plot1.png

class T_M(Data):
    def __init__(self, opt='rmsprop', number=2500):
        # 命令设置
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
        ap.add_argument("-p", "--plot", type=str, required=True, help="path to train process plot")
        self.args = vars(ap.parse_args())

        gpu_fraction = 0.9
        # GPU 加速设置，系数指定GPU内存占比
        tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        KTF.set_session(tf.Session(config=config))  # 设置GPU加速

        super().__init__(number) # 继承，Data()类初始化
        self.opt = opt
        self.number = number
        self.train_df, self.validate_df, self.train_generator,\
        self.validation_generator, self.total_train, self.total_validate = Data(self.number).getdata()


    def model_train(self): # 模型设置
        FAST_RUN = False
        epochs = 3 if FAST_RUN else 30
        INIT_LR = 1e-3  # 优化器初始学习率

        model = Sequential()
        # 第一层 卷积，池化
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT,
                                                                     self.IMAGE_CHANNELS)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # 第二层
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # 第三层
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # 全连接层
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        print(model.summary())

        # 训练参数设置
        earlystop = EarlyStopping(patience=10)  # 10步准确率没有明显变化，就提前停止
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=1e-5)  # 2步准确率没有提高就降低学习率
        callbacks = [earlystop, learning_rate_reduction]

        print("[INFO] ready to train, Train on {} smaples, Val on {} samples".format(self.total_train, self.total_validate))
        # 开始训练
        H = model.fit_generator(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.total_validate//self.batch_size,
            steps_per_epoch=self.total_train//self.batch_size,
            callbacks=callbacks
            )
        print("[INFO] serializing network to '{}'...".format(self.args["model"]))
        model.save(self.args["model"])
        return H


    def plot(self, H):
        # 记录训练误差和准确率
        plt.figure()
        actural_epochs = len(H.history['loss'])
        plt.plot(np.arange(0, actural_epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, actural_epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, actural_epochs), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, actural_epochs), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc='best')
        plt.savefig(self.args["plot"])


if __name__ == '__main__':
    start = time.perf_counter()
    H = T_M().model_train()
    T_M().plot(H)
    end = time.perf_counter()
    print("运行时间：{:.3f}s".format(end - start))


