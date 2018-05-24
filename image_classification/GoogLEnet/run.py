from model import GoogLEnet
from preprocess import prerprocess_train, prerprocess_test
import tensorflow.contrib.eager as tfe


def main():
    dir_name = "../data/"
    data_train = prerprocess_train(dir_name)
    data_test = prerprocess_test(dir_name)  # to test at once

    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    lenet_model = GoogLEnet(device_name=device)
    # lenet_model.load()  # you can load the latest model you saved
    lenet_model.fit(data_train, data_test, epochs=100, verbose=10, batch_size=32)


if __name__ == "__main__":
    main()
