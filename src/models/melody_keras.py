import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Reshape

from src.tools.data.batcher import MelodyBatcher


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

    model.compile(loss='binary_crossentropy', optimizer='adam')
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
def main(name):
    # Create batcher
    dataset_path = f'{project_dir}/data/processed/{name}'.replace('\\', '/')
    train_batcher = MelodyBatcher(f'{dataset_path}/train', buffer_size=10000, batch_count=40)
    validation_batcher = MelodyBatcher(f'{dataset_path}/validation', buffer_size=500)

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

    model.fit_generator(train_generator(), samples_per_epoch=100, epochs=2, verbose=1, callbacks=[],
                        validation_data=validation_generator(), validation_steps=10, workers=1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
