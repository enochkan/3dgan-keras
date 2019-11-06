from models import Generator, Discriminator
import numpy as np

def test(args):
    generator = Generator(args).build_generator()
    generator.load_weights(args.checkpoint_path + 'generator_' + str(args.test_epoch+1)+'.h5')
    sample_noise = np.random.normal(0, 0.33, size=[args.batch_size, 1, 1, 1, args.latent_dim]).astype(np.float32)
    generated_volumes = generator.predict(sample_noise, verbose=1)
    generated_volumes.dump(args.sample_path + '/test_images_' + str(args.test_epoch + 1))