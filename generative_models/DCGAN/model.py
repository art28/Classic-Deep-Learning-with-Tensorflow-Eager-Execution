import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os


class Generator(tf.keras.Model):
    """ Generator Module for GAN
    Args:
        noise_dim: dimension of noise z. basically 100 is used
        output_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 output_dim=(28,28,1)):
        super(Generator, self).__init__()

        self.output_dim = output_dim

        self.dense_G_1 = tf.layers.Dense(49*256, activation=tf.nn.relu)
        self.bn1 = tf.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        self.deconv1 = tf.layers.Conv2DTranspose(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.bn2 = tf.layers.BatchNormalization()
        self.deconv2 = tf.layers.Conv2DTranspose(64, (3, 3), (2, 2), padding="same", activation=tf.nn.relu)
        self.bn3 = tf.layers.BatchNormalization()
        self.deconv3 = tf.layers.Conv2DTranspose(32,(3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.bn4 = tf.layers.BatchNormalization()
        self.deconv4 = tf.layers.Conv2DTranspose(1, (3, 3), (2, 2), padding="same", activation=tf.nn.sigmoid)

    def generate(self, Z, training):
        """
        Args:
            Z : input noise
        Return:
            gen : generated image
        """
        gen = self.dense_G_1(Z)

        gen = self.bn1(gen, training)
        gen = self.reshape(gen)

        gen = self.deconv1(gen)
        gen = self.bn2(gen, training)

        gen = self.deconv2(gen)
        gen = self.bn3(gen, training)

        gen = self.deconv3(gen)
        gen = self.bn4(gen, training)

        gen = self.deconv4(gen)

        return gen

    def __call__(self, Z , training):
        return self.generate(Z, training)


class Discriminator(tf.keras.Model):
    """ Discriminator Module for GAN
        get 28*28 image
        returns 1 for real image, 0 for zero image
    Args:
        input_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 input_dim=(28,28,1)):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim

        self.conv1 = tf.layers.Conv2D(32, (3, 3), (2, 2), padding="valid", activation=tf.nn.leaky_relu)
        self.batch1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.leaky_relu)
        self.batch2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(64,(3, 3), (2, 2), padding="valid", activation=tf.nn.leaky_relu)
        self.batch3 = tf.layers.BatchNormalization()
        self.conv4 = tf.layers.Conv2D(32,(3, 3), (1, 1), padding="same", activation=tf.nn.leaky_relu)
        self.batch4 = tf.layers.BatchNormalization()
        self.conv5 = tf.layers.Conv2D(16, (3, 3), (2, 2), padding="valid", activation=tf.nn.sigmoid)
        self.batch5 = tf.layers.BatchNormalization()
        self.conv6 = tf.layers.Conv2D(1, (2, 2), (1, 1), padding="valid")

        self.flatten = tf.layers.Flatten()

    def discriminate(self, X, training):
        """
        Args:
            X : input image
        Return:
            x: sigmoid logit[0, 1]
        """
        x = self.conv1(X)

        x = self.batch1(x, training)
        x = self.conv2(x)

        x = self.batch2(x, training)
        x = self.conv3(x)

        x = self.batch3(x, training)
        x = self.conv4(x)

        x = self.batch4(x, training)
        x = self.conv5(x)

        x = self.batch5(x, training)
        x = self.conv6(x)

        x = self.flatten(x)

        return x

    def __call__(self, X, training):
        return self.discriminate(X, training)


class DCGAN(tf.keras.Model):
    """ Generative Adversarial Network model for mnist dataset.
    Args:
        noise_dim: dimension of noise z. Basically 100.
        output_dim: dimension of output image. 28*28 in MNIST.
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 noise_dim=100,
                 output_dim=(28,28,1),
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(DCGAN, self).__init__()

        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        self.generator = Generator(self.output_dim)
        self.discriminator = Discriminator(self.output_dim)

        # optimizer
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.global_step = 0
        self.epoch_loss_G = 0.
        self.epoch_loss_D = 0.

    def loss_G(self, Z, training):
        """calculate loss of generator
        Args:
            Z : noise vector
        """
        fake = self.generator(Z, training)
        logits = self.discriminator(fake, training)

        loss_val = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits)

        self.epoch_loss_G += tf.reduce_mean(loss_val)
        return loss_val

    def loss_D(self, Z, real, training):
        """calculate loss of discriminator
        Args:
            Z : noise vector
            real : real image
        """
        fake = self.generator(Z, training)
        logits_fake = self.discriminator(fake, training)
        logits_real = self.discriminator(real, training)

        loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)

        loss_val = 0.5 * tf.add(loss_fake, loss_real)

        self.epoch_loss_D += tf.reduce_mean(loss_val)
        return loss_val

    def grad_G(self, Z, training):
        """calculate gradient of the batch for generator
        Args:
            Z : noise vector
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_G(Z, training)
        return tape.gradient(loss_val, self.generator.variables)

    def grad_D(self, Z, real, training):
        """calculate gradient of the batch for discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_D(Z, real, training)
        return tape.gradient(loss_val, self.discriminator.variables)

    def grad_both(self, Z, real, training):
        """calculate gradient of the batch for both generator and discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape(persistent=True) as tape:
            loss_G = self.loss_G(Z, training)
            loss_D = self.loss_D(Z, real, training)
        return tape.gradient(loss_G, self.generator.variables), tape.gradient(loss_D, self.discriminator.variables)

    def fit(self, train_data, epochs=1, verbose=1, gen_step=3, dis_step=2, batch_size=32, saving=False):
        """train the GAN network
        Args:
            train_data: train dataset
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            gen_step & dis_step : step distribution, train would be done by [gen_step] of generator learning, and [dis_step] of both generator&discriminator learning
            batch_size : training batch size
            saving: whether to save checkpoint or not
        """

        ds = train_data.shuffle(10000).batch(batch_size)

        with tf.device(self.device_name):
            for i in range(epochs):
                self.epoch_loss_G = 0.0
                self.epoch_loss_D = 0.0

                for (X,) in ds:
                    Z = tf.random_normal((X.shape[0], self.noise_dim))

                    if (self.global_step % (gen_step + dis_step)) >= gen_step:
                        grads_G, grads_D = self.grad_both(Z, X, True)
                        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.variables))
                        self.optimizer_D.apply_gradients(zip(grads_D, self.discriminator.variables))
                    else:
                        grads_G = self.grad_G(Z, True)
                        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.variables))

                self.global_step += 1

                if i == 0 or ((i + 1) % verbose == 0):
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % (self.epoch_loss_G + self.epoch_loss_D))
                    print("GEN   loss   : %.4f" % self.epoch_loss_G)
                    print("DIS   loss   : %.4f" % self.epoch_loss_D)

                    if saving:
                        import os
                        if not os.path.exists(self.checkpoint_directory):
                            os.makedirs(self.checkpoint_directory)
                        self.save()
                    print("=" * 25)

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)

    def load(self, global_step="latest"):
        dummy_input_G = tf.zeros((1, self.noise_dim))
        dummy_input_D = tf.zeros((1,)+ self.output_dim)

        dummy_img = self.generator(dummy_input_G)
        dummy_logit = self.discriminator(dummy_input_D)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)
