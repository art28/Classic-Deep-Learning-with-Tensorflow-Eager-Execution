from model import Alexnet
from preprocess import prerprocess_train, prerprocess_test
import tensorflow.contrib.eager as tfe


def main():
    dir_name = "../data/"
    data_train = prerprocess_train(dir_name)
    data_test = prerprocess_test(dir_name).batch(10000)

    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    alex_model = Alexnet(device_name=device)
    # alex_model.load()
    alex_model.fit(data_train, data_test, epochs=50000, verbose=10, batch_size=32)

if __name__ == "__main__":
    main()