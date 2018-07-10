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


def generator_from_batcher(batcher, sample_per_batch=None):
    def generator():
        while 1:
            x, y = batcher.get_batches(sample_per_batch)
            yield x['x'], y
    return generator


@click.command()
@click.argument('name', type=click.STRING)  # Dataset name
@click.option('--model_name', default='melody', type=click.STRING)  # Dataset name
def main(name, model_name):
    # Create batcher
    dataset_path = f'data/processed/{name}'.replace('\\', '/')
    train_batcher = MelodyBatcher(f'{dataset_path}/train', buffer_size=60000, batch_count=80)
    validation_batcher = MelodyBatcher(f'{dataset_path}/validation', buffer_size=600)

    def train_generator():
        while 1:
            x, y = train_batcher.get_batches()
            yield x, y

    def validation_generator():
        while 1:
            x, y = validation_batcher.get_batches()
            yield x, y

    # Create model
    model = build_model()

    # Checkpoint
    filepath = f'models/{model_name}/weights_save.hdf5'
    os.makedirs(f'models/{model_name}', exist_ok=True)
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    model.fit_generator(train_generator(), samples_per_epoch=200, epochs=5, verbose=1, callbacks=callbacks_list, validation_data=validation_generator(),
                        validation_steps=10, workers=1)