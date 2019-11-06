from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Generator():
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        self.kernel_size = (args.kernel_size, args.kernel_size, args.kernel_size)
        self.strides = (args.strides, args.strides, args.strides)

    def build_generator(self):
        model = Sequential()
        model.add(Deconv3D(filters=512, kernel_size=self.kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=256, kernel_size=self.kernel_size,
                      strides=self.strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=128, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=64, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Deconv3D(filters=1, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.summary()

        noise = Input(shape=(1, 1, 1, self.latent_dim))
        image = model(noise)

        return Model(inputs=noise, outputs=image)

class Discriminator():
    def __init__(self, im_dim=64, kernel_size = 4, strides=2):
        self.im_dim = (im_dim, im_dim, im_dim)
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.strides = (strides, strides, strides)
        self.optim = Adam(0.0002, 0.5)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv3D(filters=64, kernel_size=self.kernel_size,
                    strides=self.strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=128, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=256, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=512, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=1, kernel_size=self.kernel_size,
                         strides=(1, 1, 1), kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.summary()

        image = Input(shape=(self.im_dim[0], self.im_dim[0], self.im_dim[0], 1))
        validity = model(image)

        return Model(inputs=image, outputs=validity)
