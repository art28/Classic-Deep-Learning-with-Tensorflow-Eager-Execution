import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import mnist
from model import VAE

tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

def main():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    ds_train = tf.data.Dataset.from_tensor_slices((x_train,))

    vae = VAE()
    vae.fit(ds_train, saving=True, epochs=500)

if __name__ == "__main__":
    main()