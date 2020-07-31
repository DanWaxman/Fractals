import multiprocessing as mp
import random

import cv2 as cv
import numpy as np
import tqdm
from PIL import Image

import cmath


class Mandelbrot:
    def __init__(self, height, width, max_it=1000, escape_z=4):
        self.height = height
        self.width = width
        self.max_it = max_it
        self.escape_z = escape_z

        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def iterate_point(self, c):
        z = 0 + 0j
        for i in range(self.max_it):
            z = z ** 2 + c
            if z.real * z.real + z.imag * z.imag > self.escape_z:
                return False

        return True

    def index2c(self, x, y):
        a = x / (self.width / 3.5) - 2.5
        b = y / (self.height / 2) - 1

        return a + b * 1j

    def thread_generate(self, bounds):
        i_min, i_max = bounds
        tmp = np.zeros((self.height, i_max - i_min, 3), dtype=np.uint8)

        for i in range(i_min, i_max):
            for j in range(self.height):
                c = self.index2c(i, j)
                result = 255 if self.iterate_point(c) else 0
                tmp[j, i - i_min] = result

        return (i_min, i_max, tmp)

    def generate_image(self):
        with mp.Pool(12) as p:
            n_jobs = 100
            i_mins = [i * self.width // n_jobs for i in range(n_jobs)]
            i_maxs = [(i + 1) * self.width // n_jobs for i in range(n_jobs)]
            args = zip(i_mins, i_maxs)
            for partial_result in tqdm.tqdm(p.imap_unordered(self.thread_generate, args), total=n_jobs):
                i_min, i_max, arr = partial_result
                self.image_data[:, i_min:i_max, :] = arr

    def get_image(self):
        return Image.fromarray(self.image_data, 'RGB')


def in_bulb(z):
    cr = z.real*z.real + z.imag*z.imag
    ci = np.sqrt(cr - z.real/2 + 0.0625)
    return (16 * cr * ci) < (5 * ci - 4 * z.real + 1)


class Buddhabrot:
    def __init__(self, height, width, points=10000, max_it=(1000, 1000, 1000), x_bounds=(-2, 1.5),
                 y_bounds=(-1.3, 1.3)):
        self.height = height
        self.width = width
        self.points = points
        self.max_it = max_it
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.float64)

    def c2index(self, c):
        x = int((c.real - self.x_bounds[0]) * self.width / (abs(self.x_bounds[0]) + abs(self.x_bounds[1])))
        y = int((c.imag - self.y_bounds[0]) * self.height / (abs(self.y_bounds[0]) + abs(self.y_bounds[1])))

        return (x, y)

    def iterate_point(self, c):
        trace = [[], [], []]
        z = c

        if in_bulb(z):
                return []

        for i in range(max(self.max_it)):
            z = z ** 2 + c

            if i < self.max_it[0]:
                trace[0].append(z)
            if i < self.max_it[1]:
                trace[1].append(z)
            if i < self.max_it[2]:
                trace[2].append(z)

            if z.real * z.real + z.imag * z.imag > 4.0:
                return trace

        return []

    def thread_generate(self, points):
        tmp = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for n in range(points):
            # TODO: Do some smarter sampling
            rand_c = random.uniform(-2, 2) + random.uniform(-2, 2) * 1j

            result = self.iterate_point(rand_c)
            for j, triplet in enumerate(result):
                for trace_z in triplet:
                    x, y = self.c2index(trace_z)
                    _, yp = self.c2index(trace_z.conjugate())

                    if 0 <= x < self.width and 0 <= y < self.height and 0 <= yp < self.height:
                        tmp[y, x, j] = tmp[y, x, j] + 1
                        tmp[yp, x, j] = tmp[yp, x, j] + 1

        return tmp

    def generate_image(self):
        with mp.Pool(12) as p:
            n_jobs = self.points // 10 ** 4
            for partial_result in tqdm.tqdm(
                    p.imap_unordered(self.thread_generate, iter([self.points // n_jobs] * n_jobs)), total=n_jobs):
                self.image_data = self.image_data + partial_result
        self.image_data = ((self.image_data / np.amax(self.image_data, axis=(0, 1))) * 255).astype(np.uint8)
        self.image_data = cv.fastNlMeansDenoisingColored(self.image_data, None, h=1, hColor=1)

    def get_image(self):
        return Image.fromarray(self.image_data.astype(np.uint8), 'RGB')

    def get_image_array(self):
        return self.image_data


class JuliaSet:
    def __init__(self, height, width, max_it=1000, escape_z=4, c=0.618 + 0.2j, verbose=True):
        self.height = height
        self.width = width
        self.max_it = max_it
        self.escape_z = escape_z
        self.c = c
        self.verbose = verbose

        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def iterate_point(self, z_0):
        z = z_0
        for i in range(self.max_it):
            z = z ** 2 + self.c
            if z.real * z.real + z.imag * z.imag > self.escape_z:
                return [i / self.max_it * 255, 255, 255]

        return [255, 255, 0]

    def index2c(self, x, y):
        a = x / (self.width / 4) - 2
        b = y / (self.height / 4) - 2

        return a + b * 1j

    def thread_generate(self, bounds):
        i_min, i_max = bounds
        tmp = np.zeros((self.height, i_max - i_min, 3), dtype=np.uint8)

        for i in range(i_min, i_max):
            for j in range(self.height):
                c = self.index2c(i, j)
                result = self.iterate_point(c)
                tmp[j, i - i_min] = result

        return (i_min, i_max, tmp)

    def generate_image(self):
        with mp.Pool(12) as p:
            n_jobs = 100
            i_mins = [i * self.width // n_jobs for i in range(n_jobs)]
            i_maxs = [(i + 1) * self.width // n_jobs for i in range(n_jobs)]
            args = zip(i_mins, i_maxs)
            for partial_result in tqdm.tqdm(p.imap_unordered(self.thread_generate, args), total=n_jobs,
                                            disable=not self.verbose):
                i_min, i_max, arr = partial_result
                self.image_data[:, i_min:i_max, :] = arr

    def get_image(self):
        return Image.fromarray(self.image_data, 'HSV')


if __name__ == '__main__':
    buddhabrot = Buddhabrot(1000, 1000, x_bounds=(-2,2), y_bounds=(-2,2), max_it=(5000,500,50), points=10**8)
    buddhabrot.generate_image()
    img = buddhabrot.get_image()
    img.show()
