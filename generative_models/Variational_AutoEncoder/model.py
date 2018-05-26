import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os

class VAE(tf.keras.Model):
    """ Variational Autoencoder model for mnist dataset.
    Args:
        input_dim: dimension of input. (28 * 28) for mnist(height * width)
        z_dim : dimension of z, which is compressed feature vector of input
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """
    def __init__(self, input_dim = 28*28,
                 z_dim = 10,
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # Encoder layers
        self.encode_dense1 = tf.layers.Dense(512, activation=tf.nn.elu)
        self.encode_dense2 = tf.layers.Dense(384, activation=tf.nn.elu)
        self.encode_dense3 = tf.layers.Dense(256, activation=tf.nn.elu)
        self.encode_mu = tf.layers.Dense(z_dim)
        self.encode_logsigma = tf.layers.Dense(z_dim)

        # Decoder layers
        self.decode_dense1 = tf.layers.Dense(256, activation=tf.nn.elu)
        self.decode_dense2 = tf.layers.Dense(384, activation=tf.nn.elu)
        self.decode_dense3 = tf.layers.Dense(512, activation=tf.nn.elu)
        self.decode_out_layer = tf.layers.Dense(self.input_dim)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.global_step = 0
        self.epoch_reconstruction_loss = 0.
        self.epoch_KL_loss = 0.
        self.epoch_loss = 0.

    def encoding(self, X):
        """encoding input data to normal distribution
        Args:
            X : input tensor
        Returns:
            mu : mean of distribution
            logsigma : log value of variation of distribution
        """
        z = self.encode_dense1(X)
        z = self.encode_dense2(z)
        z = self.encode_dense3(z)
        mu = self.encode_mu(z)
        logsigma = self.encode_logsigma(z)

        return mu, logsigma

    def sampling_z(self, z_mu, z_logsigma):
        """sampling z using mu and logsigma, using reparameterization trick
        Args:
            z_mu : mean of distribution
            z_logsigma : log value of variation of distribution
        Return:
            z value
        """
        epsilon = tf.random_normal(shape=tf.shape(z_mu), dtype=tf.float32)
        return z_mu + tf.exp(z_logsigma*0.5) * epsilon

    def decoding(self, Z):
        """image generation using z value
        Args:
            Z : z value, which is compressed feature part of the data
        Returns:
            x_decode : generated image
            sigmoid(x_decode) : generated image + sigmoid activation
        """
        x_decode = self.decode_dense1(Z)
        x_decode = self.decode_dense2(x_decode)
        x_decode = self.decode_dense3(x_decode)
        x_decode = self.decode_out_layer(x_decode)

        return x_decode, tf.nn.sigmoid(x_decode)

    def loss(self, X):
        """calculate loss of VAE model
        Args:
            X : original image batch
        """
        mu, logsigma = self.encoding(X)
        Z = self.sampling_z(mu, logsigma)
        X_decode, _ = self.decoding(Z)


        # what sigmoid_corss_entropy do
        # 1. cross entropy of [sigmoid(logits) & labels]
        # 2. mean of dimensions(input_dim)
        # 3. mean of batches
        # we only need 1 & 3 so revert 2 by multiplying input_dim
        reconstruction_loss = self.input_dim * tf.losses.sigmoid_cross_entropy(logits=X_decode, multi_class_labels=X)

        kl_div = - 0.5 * tf.reduce_sum(1. + logsigma - tf.square(mu) - tf.exp(logsigma), axis=1)


        total_loss = tf.reduce_mean(reconstruction_loss + kl_div)

        self.epoch_loss += total_loss

        self.epoch_reconstruction_loss += tf.reduce_mean(reconstruction_loss)
        self.epoch_KL_loss += tf.reduce_mean(kl_div)

        return total_loss

    def grad(self, X):
        """calculate gradient of the batch
        Args:
            X : input tensor
        """
        with tfe.GradientTape() as tape:
            loss_val = self.loss(X)
        return tape.gradient(loss_val, self.variables)

    def fit(self, train_data, epochs=1, verbose=1, batch_size=32, saving=False):
        """train the network
        Args:
            train_data: train dataset
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            batch_size : training batch size
            saving: whether to save checkpoint or not
        """

        ds = train_data.shuffle(10000).batch(batch_size)

        with tf.device(self.device_name):
            for i in range(epochs):
                self.epoch_loss = 0.0
                self.epoch_reconstruction_loss =0.
                self.epoch_KL_loss =0.
                for (X,) in ds:
                    grads = self.grad(X)
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                self.global_step += 1

                if i == 0 or ((i + 1) % verbose == 0):
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % self.epoch_loss)
                    print("RECON loss   : %.4f" % self.epoch_reconstruction_loss)
                    print("KL    loss   : %.4f" % self.epoch_KL_loss)

                    if saving:
                        import os
                        if not os.path.exists(self.checkpoint_directory):
                            os.makedirs(self.checkpoint_directory)
                        self.save()
                    print("=" * 25)

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)

    def load(self, global_step="latest"):
        dummy_input = tf.zeros((1, self.input_dim))
        dummy_mu, dummy_sigma = self.encoding(dummy_input)
        dummy_z = self.sampling_z(dummy_mu, dummy_sigma)
        dummy_ret = self.decoding(dummy_z)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = int(global_step)

        print("load %s" % self.global_step)