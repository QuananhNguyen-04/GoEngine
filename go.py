#!/usr/bin/env python
# coding: utf-8

"""Go library made with pure Python.

This library offers a variety of Go related classes and methods.

There is a companion module called 'goban' which serves as a front-end
for this library, forming a fully working go board together.

"""

import numpy as np

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Stone(object):
    def __init__(self, board, point, color):
        """Create and initialize a stone.

        Arguments:
        board -- the board which the stone resides on
        point -- location of the stone as a tuple, e.g. (3, 3)
                 represents the upper left hoshi
        color -- color of the stone

        """
        self.board = board
        self.point = point
        self.color = color
        self.group = self.find_group()

    def remove(self):
        """Remove the stone from board."""
        self.group.stones.remove(self)
        del self

    @property
    def neighbors(self):
        """Return a list of neighboring points."""
        neighboring = [
            (self.point[0] - 1, self.point[1]),
            (self.point[0] + 1, self.point[1]),
            (self.point[0], self.point[1] - 1),
            (self.point[0], self.point[1] + 1),
        ]
        # clip neighbor
        for point in neighboring:
            if not 0 < point[0] < 19 or not 0 < point[1] < 19:
                neighboring.remove(point)
        return neighboring

    @property
    def liberties(self):
        """Find and return the liberties of the stone."""
        liberties = self.neighbors
        stones = self.board.search(points=self.neighbors)
        for stone in stones:
            liberties.remove(stone.point)
        return liberties

    def find_group(self):
        """Find or create a group for the stone."""
        groups = []
        stones = self.board.search(points=self.neighbors)
        for stone in stones:
            if stone.color == self.color and stone.group not in groups:
                groups.append(stone.group)
        if not groups:
            group = Group(self.board, self)
            return group
        else:
            if len(groups) > 1:
                for group in groups[1:]:
                    groups[0].merge(group)
            groups[0].stones.append(self)
            return groups[0]

    def __str__(self):
        """Return the location of the stone, e.g. 'D17'."""
        # return 'ABCDEFGHJKLMNOPQRST'[self.point[0]-1] + str(20-(self.point[1]))
        return f"({self.point[0]},{self.point[1]})"


class Group(object):
    def __init__(self, board, stone: Stone):
        """Create and initialize a new group.

        Arguments:
        board -- the board which this group resides in
        stone -- the initial stone in the group

        """
        self.board = board
        self.board.groups.append(self)
        self.stones : list[Stone] = [stone]
        self.liberties = None
        self.color = stone.color

    def merge(self, group):
        """Merge two groups.

        This method merges the argument group with this one by adding
        all its stones into this one. After that it removes the group
        from the board.

        Arguments:
        group -- the group to be merged with this one

        """
        for stone in group.stones:
            stone.group = self
            self.stones.append(stone)
        self.board.groups.remove(group)
        del group

    def remove(self):
        """Remove the entire group."""
        while self.stones:
            self.stones[0].remove()
        self.board.groups.remove(self)
        del self

    def update_liberties(self):
        """Update the group's liberties.

        As this method will remove the entire group if no liberties can
        be found, it should only be called once per turn.

        """
        liberties = []
        for stone in self.stones:
            for liberty in stone.liberties:
                liberties.append(liberty)
        self.liberties = set(liberties)
        # print(self)
        # print(self.liberties)
        if len(self.liberties) == 0:
            self.remove()

    def __str__(self):
        """Return a list of the group's stones as a string."""
        return str([str(stone) for stone in self.stones])


class Board(object):
    def __init__(self):
        """Create and initialize an empty board."""
        self.groups: list[Group] = []
        self.next = BLACK
        self.influence = np.zeros(
            (19, 19), dtype=int
        )  # 19 * 19 array to hold self influence
        self.stone = np.zeros((19, 19), dtype=int)

    def search(self, point=None, points=[]):
        """Search the board for a stone.

        The board is searched in a linear fashion, looking for either a
        stone in a single point (which the method will immediately
        return if found) or all stones within a group of points.

        Arguments:
        point -- a single point (tuple) to look for
        points -- a list of points to be searched

        """
        stones = []
        # print("groups")
        # for sth in self.groups:
        #     print(sth)
        for group in self.groups:
            for stone in group.stones:
                if stone.point == point and not points:
                    return stone
                if stone.point in points:
                    stones.append(stone)
        return stones

    def turn(self):
        """Keep track of the turn by flipping between BLACK and WHITE."""
        if self.next == BLACK:
            self.next = WHITE
            return BLACK
        else:
            self.next = BLACK
            return WHITE

    def calculate_winner(self):
        def propagation(point, color=0):
            # print(point)
            influence = 3
            if color == 0: # BLACK
                bias = 1
            if color == 1: # WHITE
                bias = -1
            self.influence[point[1]][point[0]] = bias * (influence)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    pos = (point[1] + dy, point[0] + dx)
                    if pos[0] < 0 or pos[0] > 18 or pos[1] < 0 or pos[1] > 18 :
                        continue
                    if self.stone[pos[0]][pos[1]] != 0:
                        continue
                    # print("dx, dy", dx, dy, influence - abs(dx) - abs(dy))
                    temp_influence = bias * max(
                        (influence - abs(dx) - abs(dy)), 0
                    )
                    if self.influence[pos[0]][pos[1]] * bias <= 0:
                        self.influence[pos[0]][pos[1]] += temp_influence
                    else:
                        self.influence[pos[0]][pos[1]] = bias * max(bias * self.influence[pos[0]][pos[1]], bias * temp_influence)

        for group in self.groups:
            for stone in group.stones:
                if stone.color == BLACK:
                    self.stone[stone.point[1]][stone.point[0]] = 1
                elif stone.color == WHITE:
                    self.stone[stone.point[1]][stone.point[0]] = -1
        for group in self.groups:
            for stone in group.stones:
                if stone.color == BLACK:
                    # self.influence[stone.point[1]][stone.point[0]] = 5
                    propagation(stone.point, 0)
                if stone.color == WHITE:
                    # self.influence[stone.point[1]][stone.point[0]] = -5
                    propagation(stone.point, 1)
        print(self.influence)
        # print(self.stone)
