import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
from blocks import InceptionBlock


# eagerly (declared only once)
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)


class GoogLEnet(tf.keras.Model):
    """ GoogLEnet model for CIFAR-10 dataset.
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
        super(GoogLEnet, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        self.device_name = device_name

        # layer declaration

        # first convolution is skipped because image size is already small
        # self.conv1 = tf.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2, 2), padding="same",
        #                               activation=tf.nn.relu)
        # self.maxpool1 =tf.layers.MaxPooling2D((3,3),(2,2), padding="same")

        self.conv2 = tf.layers.Conv2D(32, (3,3), (1,1), padding="same", activation=tf.nn.relu)
        self.maxpool2 = tf.layers.MaxPooling2D((5,5), (1,1))  # this is custom pool to make size 28x28

        self.inception3a = InceptionBlock(conv11=8, reduce_conv33=12, conv33=16, reduce_conv55=2, conv55=4, convpool=2)
        self.inception3b = InceptionBlock(16, 16, 24, 4, 12, 8)
        self.maxpool3 = tf.layers.MaxPooling2D((3,3), (2,2), padding="same")

        self.inception4a = InceptionBlock(24, 12, 26, 2, 6, 8)
        self.inception4b = InceptionBlock(20, 14, 28, 3, 8, 8)
        self.inception4c = InceptionBlock(16, 16, 32, 3, 8, 8)
        self.inception4d = InceptionBlock(14, 18, 32, 4, 8, 8)
        self.inception4e = InceptionBlock(32, 20, 40, 4, 16, 16)
        self.maxpool4 = tf.layers.MaxPooling2D((3,3), (2,2), padding="same")

        self.inception5a = InceptionBlock(32, 20, 40, 4, 16, 16)
        self.inception5b = InceptionBlock(48, 24, 48, 6, 16, 16)
        self.avgpool = tf.layers.AveragePooling2D((7,7),(1,1))

        self.flatten = tf.layers.Flatten()
        self.dropout = tf.layers.Dropout(0.4)

        self.out_layer = tf.layers.Dense(self.out_dim)

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
        x = self.conv2(X)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = self.flatten(x)

        if training:
            x = self.dropout(x)

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
