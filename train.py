import logging
from dotenv import find_dotenv, load_dotenv
import importlib
import click


@click.command()
@click.argument('model_name', type=click.STRING)
@click.argument('dataset_name', type=click.STRING)
@click.argument('save_name', type=click.STRING)
@click.option('--processed', default=True, type=click.BOOL, help='Wether the dataset already is processed or not')
@click.option('--epochs', default=5, type=click.INT, help='Amount of epochs should be trained on')
@click.option('--samples_per_epoch', default=200, type=click.INT, help='Samples per epoch you want trained on')
@click.option('--batch_size', default=80, type=click.INT, help='Size of the batch that is trained on. Not too big or too small.')
@click.option('--buffer_size', default=3000, type=click.INT, help='Amount of batches that should be preloaded')
@click.option('--validation_steps', default=10, type=click.INT, help='Amount of steps that should be done while validating')
@click.option('--validation_batch_size', default=80, type=click.INT, help='Batch size for validation')
@click.option('--validation_buffer_size', default=1000, type=click.INT, help='Amount of batches to preload for validation')
def main(model_name, dataset_name, processed, save_name, epochs, samples_per_epoch, batch_size, buffer_size, validation_steps, validation_batch_size,
         validation_buffer_size):
    train_model = importlib.import_module(f'src.models.{model_name}')
    train_model.train(dataset_name, save_name, processed, samples_per_epoch, validation_steps, epochs, batch_size, buffer_size, validation_batch_size,
                      validation_buffer_size)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
