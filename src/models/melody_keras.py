import click
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dropout, Reshape
from src.tools.data.batcher import MelodyBatcher
import os


def build_model():
    model = Sequential()
    model.add(Reshape([85, 304, 1], input_shape=[85, 304]))
    # F0
    model.add(Conv2D(30, kernel_size=(46, 96), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 209), strides=(2, 1)))
    # F1
    model.add(Conv2D(60, kernel_size=(5, 1), strides=(1, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    # F2
    model.add(Conv2D(800, kernel_size=(8, 1), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    # F3
    model.add(Conv2D(2, kernel_size=(1, 1), strides=(1, 1), activation='softmax'))
    model.add(Reshape([2]))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    return model


def train(dataset_name, save_name, samples_per_epoch=200, validation_steps=10, epochs=5, batch_size=80, buffer_size=60000, validation_batch_size=80,
          validation_buffer_size=5000):
    # Create batcher
    dataset_path = f'data/processed/{dataset_name}'.replace('\\', '/')
    train_batcher = MelodyBatcher(f'{dataset_path}/train', buffer_size=buffer_size, batch_size=batch_size)
    validation_batcher = MelodyBatcher(f'{dataset_path}/validation', buffer_size=validation_buffer_size, batch_size=validation_batch_size)

    # Create model
    model = build_model()

    # Checkpoint
    filepath = f'models/{save_name}/weights_save.hdf5'
    os.makedirs(f'models/{save_name}', exist_ok=True)
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    if os.path.exists(filepath): model.load_weights(filepath)

    model.fit_generator(train_batcher.generator(),
                        samples_per_epoch=samples_per_epoch,
                        epochs=epochs, verbose=1, callbacks=callbacks_list,
                        validation_data=validation_batcher.generator(),
                        validation_steps=validation_steps,
                        workers=1)
