# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import src.tools.osu as osu
import random, os
import librosa
from shutil import copyfile
from src.utils.file import make_safe_filename


@click.command()
@click.argument('name', type=click.STRING)  # Dataset name
@click.argument('osu_path', type=click.Path(exists=True))  # Path to osu installation directory
@click.option('--map_filter', type=click.STRING, default='ar>=8 ranked=4 mode=0', help='Filter to select the candidate maps')
@click.option('--sample_count', type=click.INT, default=500, help='Amount of songs that should be in the dataset')
@click.option('--validation_size', type=click.FLOAT, default=0.1, help='Percentage of all file that should be used for validation')
def main(name, osu_path, map_filter, sample_count, validation_size):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    osu_path = osu_path.replace('\\', '/')

    db = osu.OsuDB(f'{osu_path}/osu!.db')
    maps = db.filter(map_filter)
    if len(maps) < sample_count: raise Exception('Not enough maps to create a dataset. Consider lowering the sample count.')

    contents = []
    validation_count = min(1, int(sample_count * validation_size))
    while len(contents) < sample_count:
        sample = maps[random.randint(0, len(maps) - 1)]
        if sample.set_id in contents: continue

        sample_type = 'train'
        if validation_count > 0 and random.randint(1, 100) <= 100 * validation_size:
            sample_type = 'validation'
            validation_count -= 1

        audio_path = f'{osu_path}/Songs/{sample.folder_name}/{sample.audio_file}'
        annotation_path = f'{osu_path}/Songs/{sample.folder_name}/{sample.osu_file}'
        output_path = f'{project_dir}/data/processed/{name}/{sample_type}'.replace('\\', '/')

        # Read timings
        ret = osu.models.Beatmap()
        try:
            annotation_file = open(annotation_path, 'r', encoding='utf8')
            osu.beatmap_reader.read_timing_points(annotation_file, ret)
            osu.beatmap_reader.read_hitobjects(annotation_file, ret)
        except Exception as e: continue

        # Check if audio is valid
        try:
            x, sr = librosa.load(audio_path)
            duration = len(x) / sr
        except Exception as e: continue

        downbeats = extract_beats(ret.timingpoints, ret.hitobjects, duration)

        # Save the data
        os.makedirs(output_path, exist_ok=True)

        librosa.output.write_wav(f'{output_path}/{make_safe_filename(str(sample))}.wav', x, sr)

        timing_changes = [f'{timing.offset/1000} {timing.mpb} {timing.meter}\n' for timing in ret.timingpoints if isinstance(timing, osu.models.KeyTimingPoint)]
        with open(f'{output_path}/{make_safe_filename(str(sample))}.txt', 'w') as f:
            f.write(f'{len(timing_changes)}\n')
            f.writelines(timing_changes)
            f.write(f'{len(downbeats)}\n')
            f.writelines([f'{downbeat[0]} {downbeat[1]} {downbeat[2]}\n' for downbeat in downbeats])

        print(audio_path)
        contents.append(sample.set_id)

    print(f'{name} and {osu_path}')


def extract_beats(timings, hitobjects, duration):
    ret = []
    timing_idx = 0
    hitobject_idx = 0
    timing_point = timings[timing_idx]
    time = timing_point.offset
    beat = 1
    epsilon = 20  # 20 ms is epsilon
    duration = duration * 1000

    while time < duration:
        if beat > timing_point.meter: beat = 1  # Reset beat
        if timing_idx + 1 < len(timings) and time > timings[timing_idx + 1].offset:  # Timing has changed
            timing_idx += 1
            if isinstance(timings[timing_idx], osu.models.KeyTimingPoint):
                timing_point = timings[timing_idx]
                time = timing_point.offset
                beat = 1

        # Keep track of current hitobject
        while hitobject_idx + 1 < len(hitobjects) and time + epsilon > hitobjects[hitobject_idx + 1].time:
            hitobject_idx += 1

        # Check wether a hitobject is a current beat
        is_hit = False
        if abs(hitobjects[hitobject_idx].time - time) < epsilon: is_hit = True

        ret.append((time / 1000, beat, is_hit))

        beat += 1
        time += timing_point.mpb

    return ret


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
