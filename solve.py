import sys
import cv2
import numpy as np
from operator import attrgetter
import random as rand
import heapq
from time import time

class Piece:
    def __init__(self, image, index):
        self._image   = image[:]
        self.id       = index
        self.rotation = 0

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    @property
    def size(self):
        return self.image.shape[0]

    @property
    def shape(self):
        return self.image.shape

    @property
    def image(self):
        return np.rot90(self._image, k=self.rotation)

    def rotate(self, num):
        self.rotation += num
        self.rotation %= 4

    def edge(self, num):
        return [self.image[0], self.image[:, 0], self.image[-1], self.image[:, -1]][(num + self.rotation) % 4]

class Slave:
    diffs = {}
    match_table = {}

    @classmethod
    def calc(cls, first, second, direction):
        return np.linalg.norm(first.edge((direction + 2) % 4) - second.edge(direction))

    @classmethod
    def add_diff(cls, pair, direction, diff):
        if pair not in cls.diffs:
            cls.diffs[pair] = [None for _ in range(2)]
        cls.diffs[pair][direction] = diff

    @classmethod
    def update_match_table(cls, first, second, direction):
        diff = cls.calc(first, second, direction)
        cls.add_diff((first.id, second.id), direction, diff)
        cls.match_table[first.id][(direction + 2) % 4].append((second.id, diff))
        cls.match_table[second.id][direction].append((first.id, diff))

    @classmethod
    def analyze(cls, pieces):
        for piece in pieces:
            cls.match_table[piece.id] = [[] for _ in range(4)]

        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                for direction in range(2):
                    cls.update_match_table(pieces[i], pieces[j], direction)
                    cls.update_match_table(pieces[j], pieces[i], direction)

        for piece in pieces:
            for direction in range(4):
                cls.match_table[piece.id][direction].sort(key=lambda x: x[1])

    @classmethod
    def get_diff(cls, pair, direction):
        return cls.diffs[pair][direction]

    @classmethod
    def match(cls, piece, direction):
        return cls.match_table[piece][direction][0][0]

class Sex:
    def __init__(self, mom, dad):
        self.mom = mom
        self.dad = dad

        self.min_row = self.max_row = self.min_col = self.max_col = 0

        self.kernel     = {}
        self.used       = set()
        self.candidates = []

    def is_valid(self, piece_id):
        return piece_id is not None and piece_id not in self.kernel

    def is_row_in(self, row):
        return abs(min(self.min_row, row)) + abs(max(self.max_row, row)) < self.mom.rows

    def is_col_in(self, col):
        return abs(min(self.min_col, col)) + abs(max(self.max_col, col)) < self.mom.cols

    def is_in(self, coord):
        row, col = coord
        return self.is_row_in(row) and self.is_col_in(col)

    def get_shared(self, piece_id, direction):
        mom_edge = self.mom.edge(piece_id, direction)
        dad_edge = self.dad.edge(piece_id, direction)

        if mom_edge == dad_edge:
            return mom_edge

    def add_shared(self, piece_id, coord, neighbor):
        heapq.heappush(self.candidates, (-99, (coord, piece_id), neighbor))

    def get_buddy(self, piece_id, direction):
        first = Slave.match(piece_id, direction)
        second = Slave.match(first, (direction + 2) % 4)

        if second == piece_id:
            for edge in [parent.edge(piece_id, direction) for parent in [self.mom, self.dad]]:
                if edge == first:
                    return edge

    def add_buddy(self, piece_id, coord, neighbor):
        heapq.heappush(self.candidates, (-99, (coord, piece_id), neighbor))

    def get_match(self, piece_id, direction):
        for piece, diff in Slave.match_table[piece_id][direction]:
            if self.is_valid(piece):
                return piece, diff

    def add_match(self, piece_id, coord, priority, neighbor):
        heapq.heappush(self.candidates, (priority, (coord, piece_id), neighbor))

    def add_candidate(self, piece_id, direction, coord):
        shared = self.get_shared(piece_id, direction)
        if self.is_valid(shared):
            self.add_shared(shared, coord, (piece_id, direction))
            return

        buddy = self.get_buddy(piece_id, direction)
        if self.is_valid(buddy):
            self.add_buddy(buddy, coord, (piece_id, direction))
            return

        match, priority = self.get_match(piece_id, direction)
        if self.is_valid(match):
            self.add_match(match, coord, priority, (piece_id, direction))
            return

    def update_boundaries(self, row, col):
        self.min_row = min(self.min_row, row)
        self.max_row = max(self.max_row, row)
        self.min_col = min(self.min_col, col)
        self.max_col = max(self.max_col, col)

    def get_available(self, row, col):
        boundaries = []

        if len(self.kernel) != len(self.mom.pieces):
            for direction, coord in enumerate([(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]):
                if coord not in self.used and self.is_in(coord):
                    self.update_boundaries(*coord)
                    boundaries.append((direction, coord))

        return boundaries

    def update(self, piece_id, coord):
        for direction, coord in self.get_available(*coord):
            self.add_candidate(piece_id, direction, coord)

    def grow(self, piece_id, coord, save=False):
        self.kernel[piece_id] = coord
        self.used.add(coord)
        self.update(piece_id, coord)

        if save:
            global ti, tr
            self.child().save(f'img/growth/{ti}-{tr}.ppm')
            tr += 1
            if tr > 9:
                ti += 1
                tr = 0

    def run(self, save=False):
        self.grow(rand.choice(self.mom.pieces).id, (0, 0), save)

        while len(self.candidates):
            _, (coord, piece_id), neighbor = heapq.heappop(self.candidates)

            if coord in self.used:
                continue

            if piece_id in self.kernel:
                self.add_candidate(neighbor[0], neighbor[1], coord)
                continue

            self.grow(piece_id, coord, save)

        return self

    def child(self):
        pieces = [None for _ in range(len(self.mom.pieces))]

        for piece, (row, col) in self.kernel.items():
            pieces[(row - self.min_row) * self.mom.cols + (col - self.min_col)] = self.mom.piece_by_id(piece)

        for i in range(len(pieces)):
            if pieces[i] == None:
                pieces[i] = Piece(np.zeros((self.mom.pieces[0].image.shape)), -1)

        return Chromosome(pieces, self.mom.rows, self.mom.cols)

class Chromosome:
    def __init__(self, pieces, rows, cols):
        self.pieces = pieces[:]
        self.rows   = rows
        self.cols   = cols
        self._score  = None

        self.piece_map = {piece.id: index for index, piece in enumerate(self.pieces)}

    def __getitem__(self, index):
        return self.pieces[index * self.cols:(index + 1) * self.cols]

    def __str__(self):
        return f'{self.score}'

    def __gt__(self, obj):
        if type(obj) == self.__class__:
            return self.score > obj.score

    def random(self):
        np.random.shuffle(self.pieces)
        return self

    @property
    def score(self):
        if self._score == None:
            score = 1
            for i in range(self.rows):
                for j in range(self.cols - 1):
                    score += Slave.get_diff((self[i][j].id, self[i][j + 1].id), 1)
            for i in range(self.rows - 1):
                for j in range(self.cols):
                    score += Slave.get_diff((self[i][j].id, self[i + 1][j].id), 0)

            self._score = 1 / score

        return self._score

    @property
    def piece_size(self):
        return self.pieces[0].size

    def piece_by_id(self, identifier):
        return self.pieces[self.piece_map[identifier]]

    def to_image(self):
        return np.concatenate([np.concatenate([[piece.image for piece in self.pieces][i * self.cols + j] for j in range(self.cols)], 1) for i in range(self.rows)])

    def edge(self, piece_id, direction):
        index = self.piece_map[piece_id]

        if direction == 0 and index >= self.cols:
            return self.pieces[index - self.cols].id

        if direction == 1 and index % self.cols > 0:
            return self.pieces[index - 1].id

        if direction == 2 and index < (self.rows - 1) * self.cols:
            return self.pieces[index + self.cols].id

        if direction == 3 and index % self.cols < self.cols - 1:
            return self.pieces[index + 1].id

    def mutate(self):
        if rand.random() < 0.5:
            row1 = rand.randrange(self.rows)
            row2 = rand.randrange(self.rows)
            itr1 = row1 * self.cols
            itr2 = row2 * self.cols

            self.pieces[itr1:itr1 + self.cols], self.pieces[itr2:itr2 + self.cols] = self.pieces[itr2:itr2 + self.cols], self.pieces[itr1:itr1 + self.cols]
        else:
            col1 = rand.randrange(self.cols)
            col2 = rand.randrange(self.cols)

            self.pieces[col1::self.cols], self.pieces[col2::self.cols] = self.pieces[col2::self.cols], self.pieces[col1::self.cols]

    def save(self, path):
        cv2.imwrite(path, self.to_image())

class Civilization:
    def __init__(self, image, piece_size, mutation_prob, population_size, generations, elite_size):
        self.image         = image
        self.mutation_prob = mutation_prob
        self.generations   = generations
        self.elite_size    = elite_size

        self.pieces, rows, cols = self.flatten_image(image, piece_size)
        self.population = [Chromosome(self.pieces, rows, cols).random() for _ in range(population_size)]

    def flatten_image(self, image, piece_size):
        rows, cols = image.shape[0] // piece_size, image.shape[1] // piece_size
        pieces = np.reshape([[Piece(img.astype(np.int16), i * cols + j) for j, img in enumerate(np.split(h, rows))] for i, h in enumerate(np.split(image, cols, 1))], (-1))

        return pieces, rows, cols

    def get_elites(self):
        return sorted(self.population, key=attrgetter("score"))[-self.elite_size:]

    def get_parents(self):
        return [rand.choices(self.population, weights=[chromosome.score for chromosome in self.population], k=2) for _ in range(len(self.population) - self.elite_size)]

    def get_best(self):
        return max(self.population, key=attrgetter("score"))

    def print_time(self):
        print(f'Taken: {time() - self.start} seconds')
        self.start = time()

    def run(self, save=False):
        self.start = time()
        Slave.analyze(self.pieces)
        self.print_time()

        best = None
        best_score = float("-inf")

        for generation in range(self.generations):
            
            new_population = []

            new_population.extend(self.get_elites())

            _save = save
            for mom, dad in self.get_parents():
                child = Sex(mom, dad).run(save=_save).child()
                if rand.random() < self.mutation_prob:
                    child.mutate()
                new_population.append(child)
                _save = False
            best = self.get_best()
            best_score = max(best_score, best)

            self.population = new_population

            print(f"Generation: {generation + 1} Best: {best}")
            best.save(f'img/{generation}.ppm')

        self.print_time()
        return best

if __name__ == "__main__":
    global ti, tr
    ti = tr = 0

    Civilization(image=cv2.imread(sys.argv[1]), piece_size=int(sys.argv[2]), mutation_prob=0.5, population_size=300, generations=30, elite_size=1).run().save(sys.argv[1].split('.')[0] + '_solution.ppm')