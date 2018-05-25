import tensorflow as tf
import tensorflow.contrib.eager as tfe


class VAE(tf.keras.Model):
    def __init__(self, input_dim = 28*28,
                 out_dim = 10,
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        self.device_name = device_name

        # Encoder
        self.encode_dense1 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.encode_dense2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.encode_mu = tf.layers.Dense(128)
        self.encode_logsigma = tf.layers.Dense(128)

        # Decoder
        self.decode_dense1 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.decode_dense2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.decode_out_layer = tf.layers.Dense(self.input_dim, activation=tf.nn.sigmoid)

        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)

        self.global_step = 0
        self.epoch_reconstruction_loss = 0.
        self.epoch_KL_loss = 0.
        self.epoch_loss = 0.

    def encoding(self, X):
        x = self.encode_dense1(X)
        x = self.encode_dense2(x)
        mu = self.encode_mu(x)
        logsigma = self.encode_logsigma(x)

        return mu, logsigma

    def sampling_z(self, z_mu, z_logsigma):
        epsilon = tf.random_normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(z_logsigma*0.5) * epsilon

    def decoding(self, Z):
        z = self.decode_dense1(Z)
        z = self.decode_dense2(z)
        z = self.decode_out_layer(z)

        return z

    def loss(self, X):
        mu, logsigma = self.encoding(X)
        Z = self.sampling_z(mu, logsigma)
        X_decode = self.decoding(Z)

        reconstruction_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=X_decode))
        KL_divergence_loss = tf.reduce_sum((1/2) * (tf.exp(logsigma) + tf.square(mu) - logsigma - 1), axis=-1)

        self.epoch_reconstruction_loss += reconstruction_loss
        self.epoch_KL_loss += KL_divergence_loss

        total_loss = reconstruction_loss + KL_divergence_loss

        self.epoch_loss += total_loss
        return total_loss

    def grad(self, X):
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
            self.global_step = global_step

        print("load %s" % self.global_step)