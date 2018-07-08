import math


class BeatmapEntry(object):
    __slots__ = ['artist', 'title', 'creator', 'version', 'audio_file', 'osu_file', 'folder_name', 'ranked',
                 'beatmap_id', 'set_id', 'ar', 'cs', 'hp', 'od', 'std_diffs', 'taiko_diffs', 'ctb_diffs',
                 'mania_diffs', 'time_drain', 'time_total', 'timingpoints', 'mode', 'loaded']

    @property
    def std_rating(self):
        return self.std_diffs[0] if len(self.std_diffs) else -1

    def __str__(self) -> str:
        return f'{self.artist} - {self.title} annotated by {self.creator}'


class TimingPoint(object):
    __slots__ = ['offset', 'slider_multiplayer', 'meter']

    def __init__(self) -> None:
        self.slider_multiplayer = 1


class KeyTimingPoint(TimingPoint):
    __slots__ = ['mpb']


class InheritedTimingPoint(TimingPoint):
    __slots__ = ['parent']

    @property
    def mpb(self):
        return self.parent.mpb


class Beatmap(object):
    __slots__ = ['title', 'artist', 'creator', 'version', 'id', 'set_id', 'hp', 'cs', 'od', 'slider_multiplayer',
                 'slider_tick_rate', 'timingpoints', 'hitobjects', 'ar']

    """
    :type timingpoints: list[TimingPoint]
    """

    def __init__(self) -> None:
        self.ar = 7
        self.hitobjects = []


class HitObject(object):
    __slots__ = ['pos', 'time']


class HitCircle(HitObject):
    __slots__ = ['time']


class Slider(HitObject):
    __slots__ = ['curve', 'repeat', 'px_length']

    def slider_duration(self, bm: Beatmap, tp):
        # TODO ask Gravified about inherited timing point slider velocity
        velocity = 100 * bm.slider_multiplayer / tp.slider_multiplayer / tp.mpb
        return self.px_length / velocity

    def pos_at(self, bm: Beatmap, tp, t):
        slider_duration = self.slider_duration(bm, tp)
        if t < self.time or t > self.time + slider_duration * self.repeat: return None
        t = t - self.time
        at = (t % slider_duration) / slider_duration
        r = math.floor(t / slider_duration) % 2
        return self.curve.pos_at(abs(r - at))


class Spinner(HitObject):
    __slots__ = 'end_time'
