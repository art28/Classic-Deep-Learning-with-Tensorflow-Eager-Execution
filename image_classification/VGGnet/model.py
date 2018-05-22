import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time


class VGGnet(tf.keras.Model):
    """ VGGnet model for CIFAR-10 dataset.
    Args:
        input_dim: dimension of input. (32, 32, 3) for CIFAR-10.(height - width - channel)
        out_dijm: dimension of output. 10 class for CIFAR-10
        learning_rate: for optimizer
        checkpoint_directory: checkpoint saving directory
        device_name: main device used for learning
    """

    def __init__(self,
                 input_dim=(32, 32, 3),
                 out_dim=10,
                 learning_rate=0.001,
                 checkpoint_directory="checkpoints/",
                 device_name="cpu:0"):
        super(VGGnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        self.device_name = device_name

        # layer declaration
        # use padding and restrict strides to (1,1)
        # because image is already too small to reduce more dimensions
        self.conv1a = tf.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                      activation=tf.nn.relu)
        self.conv1b = tf.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                      activation=tf.nn.relu)
        self.maxpool1 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.conv2a = tf.layers.Conv2D(16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv2b = tf.layers.Conv2D(16, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool2 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.conv3a = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv3b = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv3c = tf.layers.Conv2D(32, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool3 = tf.layers.MaxPooling2D((3, 3), (2, 2))

        self.conv4a = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv4b = tf.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv4c = tf.layers.Conv2D(64, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool4 = tf.layers.MaxPooling2D((3, 3), (2, 2), padding="same")

        self.conv5a = tf.layers.Conv2D(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv5b = tf.layers.Conv2D(128, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.conv5c = tf.layers.Conv2D(128, (1, 1), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool5 = tf.layers.MaxPooling2D((2, 2), (2, 2))

        self.flatten = tf.layers.Flatten()

        self.dense1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout1 = tf.layers.Dropout(0.5)

        self.dense2 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = tf.layers.Dropout(0.5)

        self.out_layer = tf.layers.Dense(self.out_dim, activation=tf.nn.softmax)

        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # global step
        self.global_step = 0

        # verbose logging
        self.epoch_loss = 0

    def predict(self, X, training):
        """predicting output of the network
        Args:
            X : input tensor
            training : whether apply dropout or not
        """
        x = self.conv1a(X)
        x = self.conv1b(X)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.maxpool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.maxpool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.maxpool5(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)

        x = self.dense2(x)
        if training:
            x = self.dropout2(x)

        x = self.out_layer(x)

        return x

    def loss(self, X, y, training):
        """calculate loss of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        prediction = self.predict(X, training)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
        self.epoch_loss += loss_value
        return loss_value

    def grad(self, X, y, trainig):
        """calculate gradient of the batch
        Args:
            X : input tensor
            y : target label(class number)
            training : whether apply dropout or not
        """
        with tfe.GradientTape() as tape:
            loss_value = self.loss(X, y, trainig)
        return tape.gradient(loss_value, self.variables)

    def fit(self, train_data, test_data=None, epochs=1, verbose=1, batch_size=32, saving=False):
        """train the network
        Args:
            train_data: train dataset
            test_data : test dataset for accuracy validation. if None, no validation
            epochs : training epochs
            verbose : for which step it will print the loss and accuracy (and saving)
            batch_size : training batch size
            saving: whether to save checkpoint or not
        """

        batch_shuffle_data = train_data.shuffle(100).batch(batch_size)

        with tf.device(self.device_name):
            for i in range(epochs):
                self.epoch_loss = 0.0
                for X, y in tfe.Iterator(batch_shuffle_data):
                    grads = self.grad(X, y, True)
                    self.optimizer.apply_gradients(zip(grads, self.variables))

                self.global_step += 1
                if i == 0 or ((i + 1) % verbose == 0):
                    print("[EPOCH %d / STEP %d]" % ((i + 1), self.global_step))
                    print("TRAIN loss   : %.4f" % self.epoch_loss)
                    if saving:
                        self.save()
                    if test_data:
                        self.test(test_data)
                    print("=" * 25)

    def test(self, test_data):
        with tf.device(self.device_name):
            total_loss = 0.0
            accuracy = tfe.metrics.Accuracy('train_acc')

            for X, y in tfe.Iterator(test_data):
                logits = self.predict(X=X, training=False)
                total_loss += self.loss(X, y, False)
                predictions = tf.argmax(logits, axis=1)
                accuracy(predictions, y)

            print("TEST accuracy: %.4f%%" % (100.0 * accuracy.result().numpy()))

            accuracy.init_variables()

    def save(self):
        tfe.Saver(self.variables).save(self.checkpoint_directory, global_step=self.global_step)

    def load(self, global_step="latest"):
        dummy_input = tf.constant(tf.zeros((1,) + self.input_dim))
        dummy_pred = self.predict(dummy_input, False)

        saver = tfe.Saver(self.variables)
        if global_step == "latest":
            saver.restore(tf.train.latest_checkpoint(self.checkpoint_directory))
            self.global_step = int(tf.train.latest_checkpoint(self.checkpoint_directory).split('/')[-1][1:])
        else:
            saver.restore(self.checkpoint_directory + "-" + str(global_step))
            self.global_step = global_step
