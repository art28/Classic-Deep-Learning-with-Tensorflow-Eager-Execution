import numpy as np

from matplotlib import pyplot as plt
import tensorflow.contrib.eager as tfe

# eagerly (declared only once)
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

def square_plot(data):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    if type(data) == list:
        data = np.concatenate(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tilethe filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    fig = plt.figure(figsize=(17, 17))
    plt.imshow(data[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.close();

    return fig

from model import VAE

model = VAE()
model.load()


generated_z = np.random.normal(size=(10**2, 128)).astype(np.float32)

logits  = model.decoding(generated_z)

generated_images = logits.numpy().reshape([100, 28, 28, 1]) * 255.0

fig = square_plot(generated_images)
fig.savefig('temp.png')