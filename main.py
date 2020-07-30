import random
import multiprocessing as mp
import numpy as np
from PIL import Image
import tqdm

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
            if z.real*z.real+z.imag*z.imag > self.escape_z:
                return False

        return True

    def index2c(self, x, y):
        a = x / (self.width / 3.5) - 2.5
        b = y / (self.height / 2) - 1

        return a + b * 1j

    def thread_generate(self, bounds):
        i_min, i_max = bounds
        tmp = np.zeros((self.height, i_max-i_min, 3), dtype=np.uint8)

        for i in range(i_min, i_max):
            for j in range(self.height):
                c = self.index2c(i, j)
                result = 255 if self.iterate_point(c) else 0
                tmp[j, i - i_min] = result

        return (i_min, i_max, tmp)

    def generate_image(self):
        with mp.Pool(12) as p:
            n_jobs = 100
            i_mins = [i * self.width//n_jobs for i in range(n_jobs)]
            i_maxs = [(i+1) * self.width//n_jobs for i in range(n_jobs)]
            args = zip(i_mins, i_maxs)
            for partial_result in tqdm.tqdm(p.imap_unordered(self.thread_generate, args), total=n_jobs):
                i_min, i_max, arr = partial_result
                self.image_data[:,i_min:i_max,:] = arr


    def get_image(self):
        return Image.fromarray(self.image_data, 'RGB')


class Buddhabrot:
    def __init__(self, height, width, points=10000, max_it=(1000,1000,1000), x_bounds=(-2,1.5), y_bounds=(-1.3,1.3)):
        self.height=height
        self.width=width
        self.points=points
        self.max_it=max_it
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.float64)

    def c2index(self, c):
        x = int((c.real - self.x_bounds[0]) * self.width / (abs(self.x_bounds[0]) + abs(self.x_bounds[1])))
        y = int((c.imag - self.y_bounds[0]) * self.height / (abs(self.y_bounds[0]) + abs(self.y_bounds[1])))

        return (x, y)

    def iterate_point(self, c):
        trace = [[], [], []]
        z = 0 + 0j

        for i in range(max(self.max_it)):
            z = z**2 + c
            if i < self.max_it[0]:
                trace[0].append(z)
            if i < self.max_it[1]:
                trace[1].append(z)
            if i < self.max_it[2]:
                trace[2].append(z)

            if z.real*z.real + z.imag*z.imag > 4.0:
                return trace

        return []


    def thread_generate(self, points):
        tmp = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for n in range(points):
            # TODO: Do some smarter sampling
            rand_c = random.uniform(*self.x_bounds) + random.uniform(*self.y_bounds) * 1j

            result = self.iterate_point(rand_c)
            for j, triplet in enumerate(result):
                for trace_z in triplet:
                    x, y = self.c2index(trace_z)

                    if 0 <= x < self.width and 0 <= y < self.height:
                        tmp[y, x, j] = tmp[y, x, j] + 1

        return tmp

    def generate_image(self):
        with mp.Pool(12) as p:
            for partial_result in tqdm.tqdm(p.imap_unordered(self.thread_generate, iter([self.points // 100] * 100)), total=100):
                self.image_data = self.image_data + partial_result
        # TODO: Do some denoising
        self.image_data = self.image_data / np.amax(self.image_data, axis=(0,1)) * 255

    def get_image(self):
        return Image.fromarray(self.image_data.astype(np.uint8), 'RGB')

    def get_image_array(self):
        return self.image_data

if __name__ == '__main__':
    '''red_img = Buddhabrot(500, 673, max_it=(5000,500,50), points=10000000)
    red_img.generate_image()
    img = red_img.get_image()
    img.show()'''
    mb = Mandelbrot(2286,4000,max_it=1000)
    mb.generate_image()
    img = mb.get_image()
    img.show()

