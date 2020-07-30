import random
import multiprocessing as mp
import numpy as np
from PIL import Image


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
            if np.abs(z) > self.escape_z:
                return False

        return True

    def index2c(self, x, y):
        a = x / (self.width / 3.5) - 2.5
        b = y / (self.height / 2) - 1

        return a + b * 1j

    def generate_image(self):
        for i in range(self.width):
            for j in range(self.height):
                if i % 100 == 0 and j == 0:
                    print(i)
                c = self.index2c(i, j)
                result = 255 if self.iterate_point(c) else 0
                self.image_data[j, i] = result

    def get_image(self):
        return Image.fromarray(self.image_data, 'RGB')


class Buddhabrot:
    def __init__(self, height, width, points=10000, max_it=1000, x_bounds=(-2,1.5), y_bounds=(-1.3,1.3)):
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
        trace = []
        z = 0 + 0j

        for i in range(self.max_it):
            z = z**2 + c
            trace.append(z)
            if z.real*z.real + z.imag*z.imag > 4.0:
                return trace

        return []


    def thread_generate(self, points):
        tmp = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for n in range(points):
            rand_c = random.uniform(*self.x_bounds) + random.uniform(*self.y_bounds) * 1j

            result = self.iterate_point(rand_c)
            for trace_z in result:
                x, y = self.c2index(trace_z)

                if 0 <= x < self.width and 0 <= y < self.height:
                    tmp[y, x] = tmp[y, x] + 1

        return tmp

    def generate_image(self):
        pool = mp.Pool(12)
        self.image_data = np.sum(pool.map(self.thread_generate, iter([self.points // 12] * 12)), axis=0)
        self.image_data = self.image_data / np.max(self.image_data) * 255

    def get_image(self):
        return Image.fromarray(self.image_data.astype(np.uint8), 'RGB')

    def get_image_array(self):
        return self.image_data

if __name__ == '__main__':
    red_img = Buddhabrot(500, 673, max_it=100, points=10000000)
    red_img.generate_image()
    red_img_array = red_img.get_image_array()

    green_img = Buddhabrot(500, 673, max_it=50, points=10000000)
    green_img.generate_image()
    green_img_array = green_img.get_image_array()

    blue_img = Buddhabrot(500, 673, max_it=100, points=10000000)
    blue_img.generate_image()
    blue_img_array = blue_img.get_image_array()

    img_array = np.stack([red_img_array[:,:,0], green_img_array[:,:,0], blue_img_array[:,:,0]], axis=2)
    img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
    img.show()
