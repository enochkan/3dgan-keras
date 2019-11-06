from models import Generator, Discriminator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from dataloader import data_loader
import numpy as np

def train(args):
    # optimizers
    dis_optim = Adam(lr=args.discrminator_lr, beta_1=args.beta)
    gen_optim = Adam(lr=args.generator_lr, beta_1=args.beta)

    # discrminator
    discriminator = Discriminator(args).build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optim, metrics=['accuracy'])

    # generator
    generator = Generator(args).build_generator()
    z = Input(shape=(args.latent_dim,))
    img = generator(z)

    # make discriminator not trainable
    discriminator.trainable = False
    validity = discriminator(img)

    combined = Model(input=z, output=validity)
    combined.compile(loss='binary_crossentropy', optimizer=gen_optim)

    # load data
    (X_train, _), (_, _) = data_loader.load_data()

    for epoch in range(args.num_epochs):
        idx = np.random.randint(len(X_train), size=args.batch_size)
        real = X_train[idx]

        z = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
        fake = generator.predict(z)

        eval_ = np.concatenate(real, fake)
        lab_ = np.reshape([1] * args.batch_size + [0] * args.batch_size, (-1, 1, 1, 1, 1))

        # calculate discrminator loss
        d_loss = discriminator.train_on_batch(eval_, lab_)
        z = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
        discriminator.trainable = False

        # calculate generator loss
        g_loss = combined.train_on_batch(z, np.reshape([1] * args.batch_size, (-1, 1, 1, 1, 1)))
        discriminator.trainable = True

        print('Training epoch {}/{}, d_loss: {}, g_loss: {}'.format(epoch+1,args.num_epochs,d_loss,g_loss))

        # sampling
        if epoch % args.sample_epoch == 0:
            print('Sampling...')
            sample_noise = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
            generated_volumes = generator.predict(sample_noise, verbose=1)
            generated_volumes.dump(args.sample_path + '/' + str(epoch+1))

        # save weights
        if epoch % args.save_epoch == 0:
            generator.save_weights(args.checkpoint_path + 'generator_' + str(epoch+1), True)
            discriminator.save_weights(args.checkpoint_path + 'discriminator_' + str(epoch+1), True)





