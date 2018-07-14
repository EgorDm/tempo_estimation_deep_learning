# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import random, os
from tqdm import tqdm
import src.features as features
from src.tools.data.audio_batcher import AudioFeatureBatcher
import msgpack
import msgpack_numpy as m
m.patch()


feature_identifiers = {
    'melody': features.extract_melody_cqt,
    'rhythm': features.extract_rhythm_odf,
    'bass': features.extract_bass_lfs,
    'harmony': features.extract_harmony_chroma,
}


@click.command()
@click.argument('dataset_name', type=click.STRING)
@click.argument('feature_name', type=click.STRING)
def main(dataset_name, feature_name):
    feature = feature_identifiers[feature_name]

    dataset_path = f'{project_dir}/data/raw/{dataset_name}'.replace('\\', '/')
    output_path = f'{project_dir}/data/processed/{dataset_name}'.replace('\\', '/')

    train_batcher = AudioFeatureBatcher(feature, datasets=f'{dataset_path}/validation', buffer_size=6000, batch_size=3000, repeat_dataset=False)
    progress = tqdm(total=len(train_batcher.train_data_files), desc='Creating dataset for feature')
    i = 0
    while True:
        batch = train_batcher.get_batch()
        if batch is None: break

        os.makedirs(f'{output_path}/validation', exist_ok=True)
        with open(f'{output_path}/validation/batch_{i}_{feature_name}.dat', 'wb') as f: msgpack.pack(batch, f)

        progress.update(train_batcher.cursor - progress.n)

        i += 1
    progress.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
