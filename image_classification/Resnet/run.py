from model import Resnet
from preprocess import prerprocess_train, prerprocess_test
import tensorflow.contrib.eager as tfe


# eagerly (declared only once)
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)


def main():
    dir_name = "../data/"
    data_train = prerprocess_train(dir_name)
    data_test = prerprocess_test(dir_name)  # to test at once

    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    res_model = Resnet(device_name=device)
    # res_model.load()  # you can load the latest model you saved
    res_model.fit(data_train, data_test, epochs=100, verbose=10, batch_size=32)


if __name__ == "__main__":
    main()
