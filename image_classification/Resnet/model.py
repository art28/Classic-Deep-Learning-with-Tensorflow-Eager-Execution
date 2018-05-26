import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time
from blocks import IdentitiyBlock_3, ConvolutionBlock_3


class Resnet(tf.keras.Model):
    """ Resnet model for CIFAR-10 dataset.
    Args:
        input_dim: dimension of input. (32, 32, 3) for CIFAR-10.(height - width - channel)
        out_dim: dimension of output. 10 class for CIFAR-10
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
        super(Resnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        self.device_name = device_name

        # layer declaration

        # first convolution is skipped because image size is already small
        # self.conv1 = tf.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2, 2), padding="same",
        #                               activation=tf.nn.relu)
        # self.maxpool1 =tf.layers.MaxPooling2D((3,3),(2,2), padding="same")

        self.conv2 = tf.layers.Conv2D(32, (3, 3), (1, 1), padding="same", activation=tf.nn.relu)
        self.maxpool2 = tf.layers.MaxPooling2D((5, 5), (1, 1))  # this is custom pool to make size 28x28

        self.iden3b = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])
        self.iden3c = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])
        self.iden3d = IdentitiyBlock_3([16, 16, 32], [(1, 1), (3, 3), (1, 1)])

        self.conv4a = ConvolutionBlock_3(filters=[32, 32, 64], kernel_sizes=[(1, 1), (3, 3), (1, 1), (1, 1)])
        self.iden4b = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4c = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4d = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4e = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])
        self.iden4f = IdentitiyBlock_3([32, 32, 64], [(1, 1), (3, 3), (1, 1)])

        self.conv5a = ConvolutionBlock_3(filters=[64, 64, 128], kernel_sizes=[(1, 1), (3, 3), (1, 1), (1, 1)])
        self.iden5b = IdentitiyBlock_3([64, 64, 128], [(1, 1), (3, 3), (1, 1)])
        self.iden5c = IdentitiyBlock_3([64, 64, 128], [(1, 1), (3, 3), (1, 1)])

        self.avgpool = tf.layers.AveragePooling2D((7, 7), (1, 1))

        self.flatten = tf.layers.Flatten()

        self.out_layer = tf.layers.Dense(out_dim)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

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
        x = self.conv2(X)
        x = self.maxpool2(x)

        x = self.iden3b(x, training=training)
        x = self.iden3c(x, training=training)
        x = self.iden3d(x, training=training)

        x = self.conv4a(x, training=training)
        x = self.iden4b(x, training=training)
        x = self.iden4c(x, training=training)
        x = self.iden4d(x, training=training)
        x = self.iden4e(x, training=training)
        x = self.iden4f(x, training=training)

        x = self.conv5a(x, training=training)
        x = self.iden5b(x, training=training)
        x = self.iden5c(x, training=training)

        x = self.avgpool(x)
        x = self.out_layer(self.flatten(x))

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
