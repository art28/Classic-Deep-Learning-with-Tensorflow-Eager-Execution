import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os


class Generator(tf.keras.Model):
    """ Generator Module for GAN
    Args:
        noise_dim: dimension of noise z. basically 10 is used
        output_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 output_dim=28 * 28):
        super(Generator, self).__init__()

        self.output_dim = output_dim

        self.dense_G_1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dense_G_2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.fake = tf.layers.Dense(self.output_dim, activation = tf.nn.tanh)

    def generate(self, Z):
        """
        Args:
            Z : input noise
        Return:
            gen : generated image
        """

        gen = self.dense_G_1(Z)
        gen = self.dense_G_2(gen)
        gen = self.fake(gen)

        return gen

    def __call__(self, Z):
        return self.generate(Z)


class Discriminator(tf.keras.Model):
    """ Discriminator Module for GAN
        get 28*28 image
        returns 1 for real image, 0 for zero image
    Args:
        input_dim: dimension of output image. 28 * 28 for MNIST
    """

    def __init__(self,
                 input_dim=28 * 28):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim

        self.dense_D_1 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.dropout1 = tf.layers.Dropout(0.7)
        self.dense_D_2 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = tf.layers.Dropout(0.7)
        self.dense_D_3 = tf.layers.Dense(32, activation=tf.nn.relu)
        self.discrimination = tf.layers.Dense(1, activation=tf.nn.sigmoid)

    def discriminate(self, X):
        """
        Args:
            X : input image
        Return:
            x: sigmoid logit[0, 1]
        """
        x = self.dense_D_1(X)
        x = self.dense_D_2(x)
        x = self.dropout1(x)
        x = self.dense_D_3(x)
        x = self.dropout2(x)
        x = self.discrimination(x)

        return x

    def __call__(self, X):
        return self.discriminate(X)


class GAN(tf.keras.Model):
    """ Generative Adversarial Network model for mnist dataset.
    Args:
        noise_dim: dimension of noise z. Basically 10.
        output_dim: dimension of output image. 28*28 in MNIST.
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 noise_dim=10,
                 output_dim=28 * 28,
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(GAN, self).__init__()

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

    def loss_G(self, Z):
        """calculate loss of generator
        Args:
            Z : noise vector
        """
        fake = self.generator(Z)
        logits = self.discriminator(fake)

        loss_val = -1. * tf.log(logits)
        # loss_val = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits), logits)

        self.epoch_loss_G += tf.reduce_mean(loss_val + 1e-8)
        return loss_val

    def loss_D(self, Z, real):
        """calculate loss of discriminator
        Args:
            Z : noise vector
            real : real image
        """
        fake = self.generator(Z)
        logits_fake = self.discriminator(fake)
        logits_real = self.discriminator(real)

        loss_fake = -1. * tf.reduce_mean(tf.log(1 - logits_fake + 1e-8))
        loss_real = -1. * tf.reduce_mean(tf.log(logits_real + 1e-8))

        # loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_fake), logits_fake)
        # loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_real), logits_real)

        loss_val = 0.5 * tf.add(loss_fake, loss_real)

        self.epoch_loss_D += tf.reduce_mean(loss_val)
        return loss_val

    def grad_G(self, Z):
        """calculate gradient of the batch for generator
        Args:
            Z : noise vector
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_G(Z)
        return tape.gradient(loss_val, self.generator.variables)

    def grad_D(self, Z, real):
        """calculate gradient of the batch for discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss_D(Z, real)
        return tape.gradient(loss_val, self.discriminator.variables)

    def grad_both(self, Z, real):
        """calculate gradient of the batch for both generator and discriminator
        Args:
            Z: noise vector
            real: real image
        """
        with tfe.GradientTape(persistent=True) as tape:
            loss_G = self.loss_G(Z)
            loss_D = self.loss_D(Z, real)
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
                        grads_G, grads_D = self.grad_both(Z, X)
                        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.variables))
                        self.optimizer_D.apply_gradients(zip(grads_D, self.discriminator.variables))
                    else:
                        grads_G = self.grad_G(Z)
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
        dummy_input_D = tf.zeros((1, self.output_dim))

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
