# Loads the maps
from struct import unpack_from
from typing import List

from src.tools.osu.models import BeatmapEntry, KeyTimingPoint, InheritedTimingPoint
from src.tools.osu.filter import filter_items, parse_filters


class DatabaseReader:
    def __init__(self, file):
        self.cursor = 0
        self._db = file.read()

    def read_num(self, length):
        type_map = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
        return self.__read_b(type_map[length], length)

    def read_date(self):
        ret = self.__read_b('Q', 8)
        return (ret / 10000) - 62135769600000

    def read_float(self, length):
        type_map = {4: 'f', 8: 'd'}
        return self.__read_b(type_map[length], length)

    def read_bool(self):
        return self.__read_b('b', 1) != 0x00

    def read_string(self):
        not_empty = self.__read_b('b', 1)
        if not_empty == 0x00: return ''

        length = self.__decode_leb128()
        ret = self.__read_b(str(length) + 's', length)

        try:
            return ret.decode('utf-8')
        except UnicodeDecodeError:
            print("Invalid UTF-8 string. Returning empty string.")
            return ''

    def skip(self, jump):
        self.cursor += jump

    def skip_string(self):
        not_empty = self.__read_b('b', 1)
        if not_empty == 0x00: return

        length = self.__decode_leb128()
        self.skip(length)

    def __read_b(self, val_type, length):
        value = unpack_from(val_type, self._db, self.cursor)[0]
        self.cursor += length
        return value

    def __decode_leb128(self):
        ret = shift = 0
        while True:
            byte = self.__read_b('B', 1)
            ret |= ((byte & 0x7F) << shift)
            if (byte & (1 << 7)) == 0:
                break
            shift += 7
        return ret


class OsuDB:
    version = 0
    user = 'Unknown'
    beatmaps = []

    def __init__(self, file_path):
        self.path = file_path
        self.read()

    def read(self):
        with open(self.path, 'rb') as file:
            reader = DatabaseReader(file)
            self.version = reader.read_num(4)
            print(self.version)
            reader.skip(13)
            self.user = reader.read_string()

            num_beatmaps = reader.read_num(4)
            print('Reading {}\'s database. Expecting {} maps.'.format(self.user, num_beatmaps))

            for _ in range(num_beatmaps):
                bm = read_beatmap(reader, self.version)
                if bm is not None:
                    self.beatmaps.append(bm)

            print('Loaded {}/{} maps.'.format(len(self.beatmaps), num_beatmaps))

    def filter(self, fs) -> List[BeatmapEntry]:
        filters = parse_filters(fs)
        print([str(filter) for filter in filters])
        return filter_items(self.beatmaps, filters)


def read_beatmap(reader: DatabaseReader, version):
    plan_b = reader.read_num(4) + reader.cursor
    try:
        ret = BeatmapEntry()
        ret.artist = reader.read_string()
        reader.skip_string()
        ret.title = reader.read_string()
        reader.skip_string()
        ret.creator = reader.read_string()
        ret.version = reader.read_string()
        ret.audio_file = reader.read_string()
        reader.skip_string()
        ret.osu_file = reader.read_string()
        ret.ranked = reader.read_num(1)
        reader.skip(14)

        ret.ar = reader.read_float(4)
        ret.cs = reader.read_float(4)
        ret.hp = reader.read_float(4)
        ret.od = reader.read_float(4)
        reader.skip(8)

        # diffs
        if version >= 20140609:
            ret.std_diffs = read_diff_pairs(reader)
            ret.taiko_diffs = read_diff_pairs(reader)
            ret.ctb_diffs = read_diff_pairs(reader)
            ret.mania_diffs = read_diff_pairs(reader)

        ret.time_drain = reader.read_num(4)
        ret.time_total = reader.read_num(4)
        reader.skip(4)

        # Timing Points
        n_tps = reader.read_num(4)
        ret.timingpoints = [read_timing_point(reader) for _ in range(n_tps)]

        ret.beatmap_id = reader.read_num(4)
        ret.set_id = reader.read_num(4)

        reader.skip(14)
        ret.mode = reader.read_num(1)
        reader.skip_string()
        reader.skip_string()
        reader.skip(2)
        reader.skip_string()
        reader.skip(10)

        ret.folder_name = reader.read_string()
        reader.skip(18)

        if reader.cursor != plan_b: raise Exception('Offsets are not equal. Entry corrupted?')
        return ret
    except Exception as e:
        print('Ripperoni ' + str(e))
        reader.cursor = plan_b
        return None


def read_diff_pairs(reader: DatabaseReader):
    ret = {}
    n = reader.read_num(4)
    for _ in range(n):
        reader.read_num(1)
        mod = reader.read_num(4)
        reader.read_num(1)
        rating = reader.read_float(8)
        ret[mod] = rating
    return ret


def read_timing_point(reader: DatabaseReader):
    mpb = reader.read_float(8)
    offset = reader.read_float(8)
    reader.read_bool()  # ret.inherited =
    if mpb > 0:
        ret = KeyTimingPoint()
        ret.offset = offset
    else:
        ret = InheritedTimingPoint()
        ret.slider_multiplayer = mpb / -100
        # TODO add parent
    return ret
