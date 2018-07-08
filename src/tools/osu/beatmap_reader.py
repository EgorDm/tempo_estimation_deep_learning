from src.tools.osu.models import *
from src.tools.osu.curves import *
from src.tools.osu.utils import original_type
from src.tools.osu.maths import Coordinate


def read_line(f):
    return f.readline().strip()


def read_section(f, section_tag):
    while read_line(f) != section_tag: continue
    while True:
        line = read_line(f)
        if len(line) == 0: break
        yield line


def read_attribute_section(f, ret, section_tag, attributes):
    for line in read_section(f, section_tag):
        line = line.split(':')
        if line[0] not in attributes: continue
        setattr(ret, attributes[line[0]], original_type(line[1].strip()))


def read_meta(f, ret):
    read_attribute_section(f, ret, '[Metadata]', {
        'Title': 'title',
        'Artist': 'artist',
        'Creator': 'creator',
        'Version': 'version',
        'BeatmapID': 'id',
        'BeatmapSetID': 'set_id'
    })


def read_difficulty(f, ret):
    read_attribute_section(f, ret, '[Difficulty]', {
        'HPDrainRate': 'hp',
        'CircleSize': 'cs',
        'OverallDifficulty': 'od',
        'ApproachRate': 'ar',
        'SliderMultiplier': 'slider_multiplayer',
        'SliderTickRate': 'slider_tick_rate'
    })


def read_timing_points(f, ret):
    ret.timingpoints = []
    last_ktp = None
    for line in read_section(f, '[TimingPoints]'):
        ltp = line.split(',')
        offset = float(ltp[0])
        mpb = float(ltp[1])

        if mpb > 0:
            tp = last_ktp = KeyTimingPoint()
            tp.mpb = mpb
        else:
            tp = InheritedTimingPoint()
            tp.slider_multiplayer = mpb / -100
            tp.parent = last_ktp
        tp.offset = offset
        tp.meter = int(ltp[2])
        ret.timingpoints.append(tp)


def read_hitobjects(f, ret):
    ret.hitobjects = []
    for line in read_section(f, '[HitObjects]'):
        lho = line.split(',')
        ret.hitobjects.append(parse_hitobject(lho))


def read(path, ):
    file = None
    ret = Beatmap()

    try:
        file = open(path, 'r', encoding='utf8')
        read_meta(file, ret)
        read_difficulty(file, ret)
        read_timing_points(file, ret)
        read_hitobjects(file, ret)
    except Exception as e:
        print(path)
        print(e)
        ret = None
        # raise e

    if file: file.close()
    return ret


def parse_hitobject(attrs):
    obj_type = int(attrs[3])
    pos = Coordinate(attrs[0], attrs[1])

    if obj_type & 1 > 0:
        ret = HitCircle()
    elif obj_type & 8 > 0:
        ret = Spinner()
        ret.end_time = attrs[5]
    elif obj_type & 2 > 0:
        ret = Slider()
        ret.px_length = float(attrs[7])
        ret.curve = parse_curve(pos, attrs[5], ret.px_length)
        ret.repeat = int(attrs[6])
    else:
        return None

    ret.pos = pos
    ret.time = int(attrs[2])
    return ret


CURVES = {'L': LinearType, 'P': PerfectType, 'B': BezierType, 'C': CatmullType}


def parse_curve(p, s, l):
    s = s.split('|')
    if s[0] not in CURVES: return None
    ret = CURVES[s[0]]()
    ret.distance = l

    points = [p]
    for c in s[1:]:
        points.append(parse_coordinate(c))
    ret.set_points(points)
    return ret


def parse_coordinate(s):
    return Coordinate(*s.split(':'))
