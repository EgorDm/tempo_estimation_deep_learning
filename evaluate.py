import logging
from dotenv import find_dotenv, load_dotenv
import importlib
import click


@click.command()
@click.argument('model_name', type=click.STRING)
@click.argument('sample_name', type=click.STRING)
@click.argument('save_name', type=click.STRING)
@click.option('--batch_size', default=100, type=click.INT, help='Size of the batch that is evaluated on. Not too big or too small.')
def main(model_name, sample_name, save_name, batch_size):
    train_model = importlib.import_module(f'src.models.{model_name}')
    train_model.evaluate(sample_name, save_name, batch_size)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
