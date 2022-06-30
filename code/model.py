import keras
from keras.optimizers import SGD
from create_dataset import Dataset
from assure_path import assure_path_exists

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset):
        model = keras.models.Sequential() ##创建对象Sequential顺序模型
        ##用.add()来堆叠模型
        model.add(keras.layers.Convolution2D(32, (3, 3), padding='same',
                                             input_shape=dataset.input_shape,
                                             activation='relu')) #添加卷积层

        model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))##池化层

        model.add(keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))

        model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten()) ##展平层

        model.add(keras.layers.Dropout(0.5))#Dropout层,防止过拟合，提升模型泛化能力

        model.add(keras.layers.Dense(dataset.class_num, activation='softmax'))#全连接网络层（神经元个数，激活函数），进行分类输出

        self.model = model
        self.model.summary() #打印网络结构和参数统计

    def train_model(self, dataset, batch_size=20, epoch=10):
        ##batch_size:每次梯度更新的样本数  epoch:训练模型迭代轮次
        sgd = SGD(lr=0.0007, decay=1e-6, momentum=0.9, nesterov=True)
        #.compile()来配置训练方法，（损失函数，优化器，准确率评测标准）
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #实例化一个ImageDataGenerator(图片生成器类)
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
            samplewise_center=False,  # 是否使输入数据的每个样本均值为0
            featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
            samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
            zca_whitening=False,  # 是否对输入数据施以ZCA白化
            rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
            width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
            height_shift_range=0.2,  # 同上，只不过这里是垂直
            horizontal_flip=True,  # 是否进行随机水平翻转
            vertical_flip=False)  # 是否进行随机垂直翻转
        #用fit_generator()函数进行训练，它将训练集分批载入显存
        self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                 batch_size=batch_size),
                                 #steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时记一个epoch结束
                                 steps_per_epoch=dataset.train_images.shape[0] / batch_size,
                                 #epoch：数据迭代的轮数
                                 epochs=epoch,
                                 validation_data=(dataset.valid_images, dataset.valid_labels))

    def evaluate_model(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def save_model(self, model_path):
        self.model.save(model_path)

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_dataset()
    dataset.prepare_dataset()
    model = Model()
    model.build_model(dataset)
    model.train_model(dataset)
    model.evaluate_model(dataset)
    assure_path_exists('../data/model/')
    model.save_model('../data/model/model.h5')
