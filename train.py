import h5py
import matplotlib.pyplot as plt
import time

from keras.optimizers import Adam

from model.generate_dataset import generate_load_data
from model.model import base_model
from model.triplet_loss import loss_function_maker


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


def main():
    embedding = 32
    batch_size = 50
    epochs = 30
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    train, test = generate_load_data(input_shape, embedding)

    model = base_model(input_shape, embedding)
    model.summary()

    loss = loss_function_maker(batch_size, 0.5)
    opt = Adam(learning_rate=0.0001)

    model.compile(loss=loss, optimizer=opt)
    history = model.fit(train[0], train[1],
                        batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(test[0], test[1]))

    plot_history(history)

    weight_pass = './save_weight/{}.hdf5'.format('weight')
    model.save_weights(weight_pass)


if __name__ == "__main__":
    main()
