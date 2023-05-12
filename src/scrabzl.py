import os
import numpy as np
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from itertools import chain, combinations, product, repeat
import pickle
import time
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from queue import Empty
from abc import ABC, abstractmethod


BLACK_BLOCK = ord('#')
SPACE = ord(' ')


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class Dictionary(tuple):
    def __new__ (cls, iterator):
        for word in iterator:
            if not isinstance(word, Word):
                raise ValueError(f"A Dictionary must contain Words (got {type(word)})")
        return super(Dictionary, cls).__new__(cls, iterator)

    def __init__(self, iterator):
        self.lengths = list(sorted(set(map(len, self))))
        self.by_length = {
            length: [word for word in self if len(word) == length]
            for length in self.lengths
        }
        self.n_words_match, self.words_match = self._get_word_index()

    def dump(self, language='french'):
        minlen = min(map(len, self))
        maxlen = max(map(len, self))
        filepath = f"../dictionaries/{language}_{len(self)}_{minlen}_{maxlen}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _get_word_index(self):
        count_dict = defaultdict(int)
        list_dict = defaultdict(list)
        for length in self.lengths:
            pset = powerset(range(length))
            for variable_letters in pset:
                for word in self.by_length[length]:
                    abstract_word = Word(
                        SPACE if i in variable_letters else c
                        for i, c in enumerate(word)
                    )
                    count_dict[abstract_word] += 1
                    list_dict[abstract_word].append(word)
        return count_dict, list_dict


WordSliceBase = namedtuple("WordSliceBase", [
    "horizontal",
    "vertical",
    "fixed",
    "i",
    "j",
    "start",
    "stop",
])


class WordSlice(WordSliceBase):
    def __new__(cls, arg_i, arg_j):
        if isinstance(arg_i, int) and isinstance(arg_j, tuple):
            horizontal, vertical = True, False
            fixed = arg_i
            i, j = arg_i, None
            start, stop = arg_j
        elif isinstance(arg_j, int) and isinstance(arg_i, tuple):
            horizontal, vertical = False, True
            fixed = arg_j
            i, j = None, arg_j
            start, stop = arg_i
        else:
            raise ValueError("Args of a WordSlice must be one tuple and one int")
        if start is None or stop is None:
            raise ValueError("start / stop can't be None in a WordSlice")
        return super(WordSlice, cls).__new__(cls,
            horizontal=horizontal,
            vertical=vertical,
            fixed=fixed,
            i=i,
            j=j,
            start=start,
            stop=stop,
        )

    def __getnewargs_ex__(self):
        arg_i = self.i if self.horizontal else (self.start, self.stop)
        arg_j = (self.start, self.stop) if self.horizontal else self.j
        return ((arg_i, arg_j), {})

    def __call__(self, a):
        bound = self.stop - self.start
        if a > bound:
            raise ValueError(f"Index out of bound {a} > {bound}")
        if self.horizontal:
            return self.i, a + self.start
        elif self.vertical:
            return a + self.start, self.j
        else:
            raise RuntimeError("WordSlice is neither horizontal nor vertical")

    def __iter__(self):
        self._a = 0
        return self

    def __next__(self):
        if self._a < self.stop - self.start:
            ret = self(self._a)
            self._a += 1
            return ret
        else:
            raise StopIteration

    def in_slice(self, i, j):
        if self.horizontal:
            return self.start <= j < self.stop and i == self.i
        elif self.vertical:
            return self.start <= i < self.stop and j == self.j

    def __repr__(self):
        hvl = 'horizontal' if self.horizontal else 'vertical' if self.vertical else 'letter'
        coord = f'{self.i} {self.start}-{self.stop}' if self.horizontal else f'{self.start}-{self.stop} {self.j}' if self.vertical else f'{self.i} {self.j}'
        return f'{hvl} {coord}'

    def __str__(self):
        return self.__repr__()


class Word(tuple):
    def __new__(cls, iterator):
        if isinstance(iterator, str):
            return super(Word, cls).__new__(cls, (ord(c) for c in iterator))
        else:
            return super(Word, cls).__new__(cls, iterator)

    def incomplete(self):
        return any(c == SPACE for c in self)

    def n_spaces(self):
        return len(tuple(c == SPACE for c in self))

    def __lt__(self, other):
        if len(self) < len(other): return True
        if len(self) > len(other): return False
        for a, b in zip(self, other):
            if a < b: return True
            if a > b: return False
        return True

    def copy(self):
        return Word(self)

    def __repr__(self):
        return f"|{''.join(chr(c) for c in self)}|"

    def __str__(self):
        return self.__repr__()


class Grid:
    def __init__(self, string):
        string = string.strip('\n')
        sub_strings = [s.strip('\n') for s in string.split('\n')]
        self.vsize = len(sub_strings)
        self.hsize = len(sub_strings[0])
        self.array = np.ndarray(shape=(self.vsize, self.hsize), dtype=np.uint8)
        for i, s in enumerate(sub_strings):
            self.array[i] = [ord(c) for c in s]
        self.slices = self._define_slices()
        self.possibility_grid = defaultdict(set)

    @classmethod
    def from_array(cls, array, slices=None, possibility_grid=None):
        grid = Grid("")
        grid.array = array
        grid.vsize = array.shape[0]
        grid.hsize = array.shape[1]
        grid.slices = grid._define_slices() if slices is None else slices
        grid.possibility_grid = defaultdict(set) if possibility_grid is None else possibility_grid
        return grid

    def copy(self):
        return Grid.from_array(
            self.array.copy(),
            slices=self.slices,
            possibility_grid=self.possibility_grid,
        )

    def _define_slices(self):
        slices = []
        if self.array.size == 0:
            return
        for i, line in enumerate(self.array):
            prev = 0
            for j, c in enumerate(line):
                if c == BLACK_BLOCK:
                    if prev != j:
                        slices.append(WordSlice(i, (prev, j)))
                    prev = j + 1
            if prev != self.hsize:
                slices.append(WordSlice(i, (prev, self.hsize)))

        for j, line in enumerate(self.array.T):
            prev = 0
            for i, c in enumerate(line):
                if c == BLACK_BLOCK:
                    if prev != i:
                        slices.append(WordSlice((prev, i), j))
                    prev = i + 1
            if prev != self.vsize:
                slices.append(WordSlice((prev, self.vsize), j))
        return slices

    def is_full(self):
        return (self.array != SPACE).all()

    def get_word_slices(self, i, j):
        return tuple(s for s in self.slices if s.in_slice(i, j))

    def __hash__(self):
        return int.from_bytes(self.array.tobytes(), 'little')

    def __setitem__(self, item, val):
        if isinstance(item, tuple):
            letter_index = item
            self.array[letter_index] = val
            for letter_index_ in chain.from_iterable(self.get_word_slices(*letter_index)):
                letter = self.array[letter_index_]
                self.possibility_grid[letter_index_] = {letter} if letter != SPACE else set()
        else:
            raise NotImplementedError('TODO?')

    def __getitem__(self, word_slice):
        if type(word_slice) is tuple:
            return self.array[word_slice]
        elif type(word_slice) is WordSlice:
            if word_slice.horizontal:
                return Word(self.array[word_slice.i, slice(word_slice.start, word_slice.stop)])
            elif word_slice.vertical:
                return Word(self.array[slice(word_slice.start, word_slice.stop), word_slice.j])
            else:
                return self.array[word_slice.i, word_slice.j]
        else:
            raise ValueError(f"Grid index must be either tuple of ints or WordSlices (got {word_slice} of type {type(word_slice)})")

    def letter_coords(self, func=lambda: True):
        for letter_coord in product(range(self.vsize), range(self.hsize)):
            if func(self[letter_coord]): yield letter_coord

    def coord_most_constrained_letter(self, dictionary):
        coord = None
        min_possibilities = 10000000000
        for letter_coord in self.letter_coords(lambda l: l == SPACE):
            possible_letters = self.get_possible_letters(dictionary, letter_coord)
            possibilities = len(possible_letters)
            if possibilities < min_possibilities:
                coord = letter_coord
                min_possibilities = possibilities
        return coord

    def get_possible_letters(self, dictionary, letter_coord):
        possible_letters = self.possibility_grid[letter_coord]
        missing = not len(possible_letters)
        if missing:
            i, j = letter_coord
            letter = self[i, j]
            if letter == SPACE: # if the grid value at (i,j) is not determined yet
                # get the slices describing the vertical and horizontal words
                # intersecting at (i,j)
                word_slices = self.get_word_slices(i, j)
                hslice = next(s for s in word_slices if s.horizontal)
                vslice = next(s for s in word_slices if s.vertical)
                # intersection at the ii'th letter in vertical word
                ii = hslice.fixed - vslice.start
                # intersection at the jj'th letter in horizontal word
                jj = vslice.fixed - hslice.start
                possible_letters = set.intersection(
                    # all letters possible at (i,j) according to the vertical word
                    {w[ii] for w in dictionary.words_match[self[vslice]]},
                    # all letters possible at (i,j) according to the horizontal word
                    {w[jj] for w in dictionary.words_match[self[hslice]]},
                )
            else:
                possible_letters = {letter} # a set containing the letter at (i,j) in the grid
            self.possibility_grid[letter_coord] = possible_letters
        return possible_letters

    def __repr__(self):
        topline = '┌' + '─' * self.array.shape[1] + '┐'
        bottomline = '└' + '─' * self.array.shape[1] + '┘'
        body = '\n'.join(
            f"│{''.join('▉' if c == BLACK_BLOCK else chr(c) for c in line)}│"
            for line in self.array
        )
        return topline + "\n" + body + "\n" + bottomline

    def __str__(self):
        return self.__repr__()


class Backtracker:
    def __init__(self, dictionary, flag_terminate=None):
        self._dictionary = dictionary
        self._level = -1
        self._visited = set()
        self.flag_terminate = flag_terminate

    @contextmanager
    def level_up(self):
        self._level += 1
        yield self._level
        self._level -= 1

    def __call__(self, grid, max_depth=None):
        if self.flag_terminate is not None and self.flag_terminate.is_set(): return
        if hash(grid) in self._visited: return
        else: self._visited.add(hash(grid))
        if grid.is_full():
            yield grid.copy()
            return

        with self.level_up() as level:
            # get the next letter to search on in the grid
            letter_coord = grid.coord_most_constrained_letter(self._dictionary)
            possible_letters = grid.get_possible_letters(self._dictionary, letter_coord)
            # If at this location, there are no possible letters
            # we arrived at a dead end
            if len(possible_letters) == 0: return
            # try each possible letter
            for letter in possible_letters:
                grid[letter_coord] = letter
                if level == max_depth:
                    yield grid.copy()
                else:
                    next_max_depth = None if max_depth is None else max_depth - 1
                    for g in self(grid, max_depth=next_max_depth): yield g
            grid[letter_coord] = SPACE # reset the grid


class SolverProcess(mp.Process):
    def __init__(self, dictionary, communication, n_solutions):
        super(SolverProcess, self).__init__()
        self.com = communication
        self.n_solutions = n_solutions
        self.busy = mp.Event()
        self.idle = mp.Event()
        self.backtrack = Backtracker(dictionary, mp.Event())

    def run(self):
        try:
            while not self.com.flag_terminate.is_set():
                if not self.com.task_queue.empty():
                    self.busy.set()
                    self.idle.clear()
                    input_grid = self.com.task_queue.get()
                    if self.n_solutions == 1:
                        output_grid = next(self.backtrack(input_grid), None)
                        if output_grid is not None: self.com.result_queue.put(output_grid)
                    else:
                        limit = repeat(None) if self.n_solutions is None else range(self.n_solutions)
                        for _, output_grid in zip(limit, self.backtrack(input_grid)):
                            self.com.result_queue.put(output_grid)
                            if self.com.flag_terminate.is_set():
                                self.busy.clear()
                                self.idle.set()
                                return
                else:
                    self.busy.clear()
                    self.idle.set()
                    self.com.flag_terminate.wait(timeout=0.5)
            self.busy.clear()
            self.idle.set()
        except KeyboardInterrupt:
            print(f"{self.name} caught KeyboardInterrupt")
            self.idle.set()


class SolverCommunication:
    def __init__(self):
        self.flag_terminate = mp.Event()
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.task_queue.cancel_join_thread()
        self.result_queue.cancel_join_thread()

class ParallelSolver(ABC):
    def __init__(self, dictionary, n_processes, n_solutions=None):
        self.n_solutions = n_solutions
        self.com = SolverCommunication()
        self.solvers = tuple(SolverProcess(
            dictionary,
            self.com,
            n_solutions,
        ) for _ in range(n_processes))

    @abstractmethod
    def solution_iterator(self, grids):
        pass

    def __enter__(self):
        for s in self.solvers:
            s.start()
        return self

    def __exit__(self, type, value, traceback):
        for s in self.solvers: s.backtrack.flag_terminate.set()
        self.com.flag_terminate.set()
        for s in self.solvers: s.join()
        for s in self.solvers: s.close()


class ParallelSolverEachOne(ParallelSolver):
    def solution_iterator(self, grids):
        for input_grid in grids: self.com.task_queue.put(input_grid)
        # wait until one of the tasks has been picked up by a solver
        while not any(s.busy.is_set() for s in self.solvers):
            for s in self.solvers: s.busy.wait(0.1)

        while any(s.busy.is_set() for s in self.solvers):
            try:
                yield self.com.result_queue.get(timeout=0.1)
            except Empty:
                pass
            except KeyboardInterrupt:
                print("ParallelSolver caught KeyboardInterrupt")
                for s in self.solvers:
                    while s.is_alive():
                        time.sleep(0.1)
                return
        time.sleep(0.1)
        while not self.com.result_queue.empty():
            yield self.com.result_queue.get()



class ParallelSolverSplit(ParallelSolver):
    def __init__(self, dictionary, n_processes, n_solutions=None):
        super(ParallelSolverSplit, self).__init__(dictionary, n_processes, n_solutions)
        self.backtrack = Backtracker(dictionary)

    def solution_iterator(self, grids, parralelization_depth=2):
        for input_grid in grids:
            # fill the task queue with the sub grids
            for sub_input_grid in self.backtrack(input_grid, max_depth=parralelization_depth):
                self.com.task_queue.put(sub_input_grid)
            # wait until one of the tasks has been picked up by a solver
            while not any(s.busy.is_set() for s in self.solvers):
                for s in self.solvers: s.busy.wait(0.1)
            # yield atmost n_solutions results for the current input_grid
            remaining = self.n_solutions or np.inf
            while remaining > 0 and any(s.busy.is_set() for s in self.solvers):
                try:
                    yield self.com.result_queue.get(timeout=0.1)
                    remaining -= 1
                except Empty:
                    pass
                except KeyboardInterrupt:
                    print("ParallelSolver caught KeyboardInterrupt")
                    for s in self.solvers:
                        s.com.flag_terminate.set()
                        s.backtrack.flag_terminate.set()
                        s.terminate()
                        s.join()
                    return
            # flush the task queue
            while not self.com.task_queue.empty():
                self.com.task_queue.get()
            # pause the solvers
            for s in self.solvers: s.backtrack.flag_terminate.set()
            for s in self.solvers: s.idle.wait()
            for s in self.solvers: s.backtrack.flag_terminate.clear()


def random_empty_grids(dictionary, vsize, hsize, n):
    min_word_len = dictionary.lengths[0]
    array = np.full(shape=(vsize, hsize), fill_value=SPACE, dtype=np.uint8)
    is_possible = (vsize * hsize) ** 2
    while np.count_nonzero(array == BLACK_BLOCK) < n and is_possible:
        is_possible -= 1
        i = np.random.randint(vsize)
        j = np.random.randint(hsize)

        right = not (array[i, j + 2:j + min_word_len + 1] == BLACK_BLOCK).any()
        bottom = not (array[i + 2:i + min_word_len + 1, j] == BLACK_BLOCK).any()
        left = True if j == 0 else not (array[i, j - min_word_len:j - 1] == BLACK_BLOCK).any()
        top = True if i == 0 else not (array[i - min_word_len:i - 1, j] == BLACK_BLOCK).any()

        right = right and (j == hsize - 1 or j < hsize - min_word_len)
        bottom = bottom and (i == vsize - 1 or i < vsize - min_word_len)
        left = left and (j == 0 or j >= min_word_len)
        top = top and (i == 0 or i >= min_word_len)

        if all((right, bottom, left, top)):
            array[i, j] = BLACK_BLOCK
    return Grid.from_array(array)


if __name__ == '__main__':
    # dictionary = Dictionary.load("../dictionaries/french_10653_2_7.pkl")
    # dictionary = Dictionary.load("../dictionaries/french_20161_2_10.pkl")

    # grids = (random_empty_grids(dictionary, 7, 10, 12) for i in range(20))
    # with ParallelSolverEachOne(dictionary, n_processes=8, n_solutions=1) as p:
    #     for grid in p.solution_iterator(grids):
    #         print(grid)

    # grids = (random_empty_grids(dictionary, 8, 8, 12) for i in range(20))
    # with ParallelSolverSplit(dictionary, n_processes=8, n_solutions=1) as p:
    #     for grid in p.solution_iterator(grids):
    #         print(grid)

    import argparse

    parser = argparse.ArgumentParser(description='Generate grids of words')
    parser.add_argument('grid_height', metavar='grid-height', type=int,
                        help='The height of the grid')
    parser.add_argument('grid_width', metavar='grid-width', type=int,
                        help='The width of the grid')
    parser.add_argument('n_blocks', metavar='n-blocks', type=int,
                        help='The number of blocks in the grid')
    parser.add_argument('dictionary_path', metavar='dictionary-path', type=str,
                        help='The path to the dictionary file')
    parser.add_argument('--n-solutions', type=int, default=None,
                        help='The number of solutions to find (default: all)')
    parser.add_argument('--n-grids', type=int, default=1,
                        help='The number of grids to generate and solve (default: 1)')
    parser.add_argument('--n-processes', type=int, default=8,
                        help='The number of processes to use (default: 8)')
    parser.add_argument('--display-freq', type=int, default=10,
                        help='Time in sec between logs (default: 10)')
    parser.add_argument('--no-display', dest='display', action='store_false',
                        help='Do not display the solutions (default: display)')

    args = parser.parse_args()

    dictionary = Dictionary.load(args.dictionary_path)

    grids = tuple(
        random_empty_grids(dictionary, args.grid_height, args.grid_width, args.n_blocks)
        for i in range(args.n_grids)
    )

    for grid in grids: print(grid.copy())

    t0 = time.time()
    t_log = 0
    with ParallelSolverSplit(
            dictionary,
            n_processes=args.n_processes,
            n_solutions=args.n_solutions) as p:
        for i, grid in enumerate(p.solution_iterator(grids, parralelization_depth=6)):
            t_now = time.time()
            if t_now - t_log > args.display_freq:
                t_log = t_now
                print(f'speed: {(i + 1) / (t_now - t0):.3f} grid/sec')
            if args.display:
                print(grid)
