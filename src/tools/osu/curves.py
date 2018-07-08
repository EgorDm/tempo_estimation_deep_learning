from src.tools.osu.maths import bezier_curve, catmull_chain, perfect_curve, Coordinate
from src.tools.osu.utils import binary_search
import numpy.linalg as linal
import numpy as np
import math


class CurveType(object):
    __slots__ = ['points', 'cum_length', 'distance']

    def __init__(self) -> None:
        self.cum_length = []
        self.points = []

    def set_points(self, ps):
        self.points = ps

    def __calc_cum_len(self):
        l = 0
        self.cum_length = [l]
        for i in range(len(self.points) - 1):
            diff = self.points[i + 1] - self.points[i]
            d = linal.norm(diff)

            if self.distance - l < d:
                l = self.distance
                self.cum_length.append(l)
                break
            l += d
            self.cum_length.append(l)

    def __progress_to_distance(self, t: float):
        return np.clip(t, 0.0, 1.0) * self.distance

    def __interpolate_vertices(self, i, d):
        if len(self.points) == 0: return Coordinate(0, 0)

        if i < 0:
            return self.points[0]
        elif i >= len(self.points):
            return self.points[-1]

        p0 = self.points[i - 1]
        p1 = self.points[i]

        d0 = self.cum_length[i - 1]
        d1 = self.cum_length[i]

        if math.isclose(d0, d1):
            return p0

        w = (d - d0) / (d1 - d0)
        return p0 + (p1 - p0) * w

    def __index_of_distance(self, d):
        i = binary_search(self.cum_length, d)
        if i < 0: i = ~i
        return i

    def pos_at(self, t):
        if len(self.cum_length) == 0: self.__calc_cum_len()

        d = self.__progress_to_distance(t)
        return self.__interpolate_vertices(self.__index_of_distance(d), d)

    def length(self):
        l = 0
        for i in range(1, len(self.points)):
            l += linal.norm(np.array(self.points[i]) - np.array(self.points[i - 1]))
        return l


class LinearType(CurveType): pass


class PerfectType(CurveType):
    def set_points(self, ps): self.points = perfect_curve(ps)


class BezierType(CurveType):
    def set_points(self, ps): self.points = bezier_curve(ps)


class CatmullType(CurveType):
    def set_points(self, ps):
        self.points = catmull_chain(ps, 0.05) if len(ps) >= 4 else bezier_curve(ps)
