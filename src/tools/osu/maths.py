from typing import List
import numpy as np
import math


class Coordinate(np.ndarray):
    @staticmethod
    def __new__(cls, x, y):
        obj = np.array([x, y], dtype=float).view(cls)
        return obj

    @property
    def x(self): return self[0]

    @x.setter
    def x(self, value): self[0] = value

    @property
    def y(self): return self[1]

    @y.setter
    def y(self, value): self[1] = value


tolerance = 0.25
tolerance_sq = tolerance * tolerance


def bezier_curve(ps: List[Coordinate]):
    control_points = ps
    count = len(control_points)

    subdivisionBuffer1 = [None] * count
    subdivisionBuffer2 = [None] * (count * 2 - 1)

    def is_flat_enough(cps: List[np.ndarray]):
        for i in range(1, len(cps) - 1):
            if math.pow(np.linalg.norm(cps[i - 1] - 2 * cps[i] + cps[i + 1]),
                        2) > tolerance_sq * 4:
                return False
        return True

    def subdivide(cps: List[np.ndarray], l: List[np.ndarray], r: List[np.ndarray]):
        midpoints = subdivisionBuffer1
        for i in range(count):
            midpoints[i] = cps[i]

        for i in range(count):
            l[i] = midpoints[0]
            r[count - i - 1] = midpoints[count - i - 1]
            for j in range(count - i - 1):
                midpoints[j] = (midpoints[j] + midpoints[j + 1]) / 2

    def approximate(cps: List[np.ndarray], output: List[np.ndarray]):
        l = subdivisionBuffer2
        r = subdivisionBuffer1

        subdivide(cps, l, r)

        for i in range(count - 1):
            l[count + i] = r[i + 1]

        output.append(cps[0])
        for i in range(1, count - 1):
            index = 2 * i
            p = 0.25 * (l[index - 1] + 2 * l[index] + l[index + 1])
            output.append(p)

    ret = []

    if count <= 1:
        return ret

    toFlatten = []
    freeBuffers = []

    toFlatten.append(control_points.copy())
    leftChild = subdivisionBuffer2

    while len(toFlatten) > 0:
        parent = toFlatten.pop()

        if is_flat_enough(parent):
            approximate(parent, ret)
            freeBuffers.append(parent)
            continue

        if len(freeBuffers) > 0:
            rightChild = freeBuffers.pop()
        else:
            rightChild = [None] * count

        subdivide(parent, leftChild, rightChild)
        for i in range(count):
            parent[i] = leftChild[i]

        toFlatten.append(rightChild)
        toFlatten.append(parent)
    ret.append(control_points[-1])
    return ret


catmull_alpha = 0.5


def catmull_point(ps: List[Coordinate], t: float) -> Coordinate:
    assert (len(ps) == 4)
    p1, p2, p3, p4 = [np.array(p, dtype=float) for p in ps]
    alpha = catmull_alpha

    return alpha * ((-p1 + 3 * p2 - 3 * p3 + p4) * t ** 3
                    + (2 * p1 - 5 * p2 + 4 * p3 - p4) * t ** 2
                    + (-p1 + p3) * t + 2 * p2)


def catmull_curve(ps: List[Coordinate], interval: int = 0.1) -> List[Coordinate]:
    return [catmull_point(ps, t) for t in np.arange(0.0, 1.0, interval)]


def catmull_chain(ps: List[Coordinate], interval: float = 0.1) -> List[Coordinate]:
    n = len(ps)
    ret = []
    for i in range(n - 3):
        ret.extend(catmull_curve(ps[i:i + 4], interval))
    return ret


def perfect_curve(ps: List[Coordinate]):
    assert len(ps) == 3
    a, b, c = ps

    aSq = np.linalg.norm(b - c) ** 2
    bSq = np.linalg.norm(a - c) ** 2
    cSq = np.linalg.norm(a - b) ** 2

    if math.isclose(aSq, 0) or math.isclose(bSq, 0) or math.isclose(cSq, 0):
        return [ps[0]]

    s = aSq * (bSq + cSq - aSq)
    t = bSq * (aSq + cSq - bSq)
    u = cSq * (aSq + bSq - cSq)


    sum = s + t + u
    if math.isclose(sum, 0): return [ps[0]]

    centre = (s * a + t * b + u * c) / sum
    dA = a - centre
    dC = c - centre

    r = np.linalg.norm(dA)

    thetaStart = math.atan2(dA[1], dA[0])
    thetaEnd = math.atan2(dC[1], dC[0])

    while thetaEnd < thetaStart:
        thetaEnd += 2 * math.pi

    dir = 1
    thetaRange = thetaEnd - thetaStart

    orthoAtoC = c - a
    orthoAtoC = (orthoAtoC[1], -orthoAtoC[0])
    if np.dot(orthoAtoC, b - a) < 0:
        dir = -dir
        thetaRange = 2 * math.pi - thetaRange

    if 2 * r <= 0.1:
        amountPoints = 2
    else:
        amountPoints = max([2, math.ceil(thetaRange / (2 * math.acos(1 - 0.1 / r)))])

    ret = []
    for i in range(amountPoints):
        fract = i / (amountPoints - 1)
        theta = thetaStart + dir * fract * thetaRange
        o = np.array((math.cos(theta), math.sin(theta))) * r
        ret.append(centre + o)

    return ret
