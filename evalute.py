import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.datasets import mnist
from sklearn.manifold import TSNE

from model.generate_dataset import generate_load_data
from model.model import base_model


def main():
    input_shape = (28, 28, 1)
    embedding = 32
    weight_pass = './save_weight/{}.hdf5'.format('weight') 

    _, (x_test, y_test) = generate_load_data(input_shape, embedding, train_mode=False)

    model = base_model(input_shape, embedding)
    model.load_weights(weight_pass)
    pred = model.predict(x_test)

    tsne = TSNE()

    tsne_train = tsne.fit_transform(pred)

    plt.figure(figsize=(16, 16))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']

    for i, c in zip(range(10), colors):
        plt.scatter(tsne_train[y_test == i, 0], tsne_train[y_test == i, 1], c=c, label=str(i))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
