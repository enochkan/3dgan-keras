from models import Generator, Discriminator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from dataloader import DataLoader
import os
import numpy as np

def train(args):
    # optimizers
    dis_optim = Adam(lr=args.discriminator_lr, beta_1=args.beta)
    gen_optim = Adam(lr=args.generator_lr, beta_1=args.beta)

    # discrminator
    discriminator = Discriminator(args).build_discriminator()
    print('Discriminator...')
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optim)

    # generator
    generator = Generator(args).build_generator()
    print('Generator')
    generator.summary()
    z = Input(shape=(1, 1, 1, args.latent_dim))
    img = generator(z)

    # make discriminator not trainable
    discriminator.trainable = False
    validity = discriminator(img)

    combined = Model(input=z, output=validity)
    combined.compile(loss='binary_crossentropy', optimizer=gen_optim)

    # load data
    data_loader = DataLoader(args)
    X_train = np.array(data_loader.load_data()).astype(np.float32)

    for epoch in range(args.num_epochs):
        #sample a random batch
        idx = np.random.randint(len(X_train), size=args.batch_size)
        # print('Sampling indices...' + str(idx))
        real = X_train[idx]

        z = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
        fake = generator.predict(z)

        real = np.expand_dims(real, axis=4)
        # eval_ = np.concatenate((real, fake))

        lab_real = np.reshape([1] * args.batch_size, (-1, 1, 1, 1, 1))
        lab_fake = np.reshape([0] * args.batch_size, (-1, 1, 1, 1, 1))
        # print(lab_real.shape)


        # calculate discrminator loss
        d_loss_real = discriminator.train_on_batch(real, lab_real)
        d_loss_fake = discriminator.train_on_batch(fake, lab_fake)

        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
        discriminator.trainable = False

        # calculate generator loss
        g_loss = combined.train_on_batch(z, np.reshape([1] * args.batch_size, (-1, 1, 1, 1, 1)))
        discriminator.trainable = True

        print('Training epoch {}/{}, d_loss_real: {}, g_loss: {}'.format(epoch+1,args.num_epochs,d_loss,g_loss))

        # sampling
        if epoch % args.sample_epoch == 0:
            if not os.path.exists(args.sample_path):
                os.makedirs(args.sample_path)
            print('Sampling...')
            sample_noise = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
            generated_volumes = generator.predict(sample_noise, verbose=1)
            generated_volumes.dump(args.sample_path + '/sample_epoch_' + str(epoch+1) + '.npy')

        # save weights
        if epoch % args.save_epoch == 0:
            if not os.path.exists(args.checkpoints_path):
                os.makedirs(args.checkpoints_path)
            generator.save_weights(args.checkpoints_path + '/generator_epoch_' + str(epoch+1), True)
            discriminator.save_weights(args.checkpoints_path + '/discriminator_epoch_' + str(epoch+1), True)





