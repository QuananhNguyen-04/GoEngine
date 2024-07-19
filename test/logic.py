import numpy as np
import init
__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

class Stone():
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
        # if len(self.liberties) == 0:
        #     board.turn()
        #     self.remove()
        #     return
    def remove(self):
        """Remove the stone from board."""
        if self in self.group.stones:
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
            if not 0 <= point[0] < 19 or not 0 <= point[1] < 19:
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
            self.board.stones[self.stones[0].point[1]][self.stones[0].point[0]] = 0
            self.stones[0].remove()
        if self in self.board.groups:
            self.board.groups.remove(self)
        del self

    def update_liberties(self) -> None:
        """Update the group's liberties.
        As this method will remove the entire group if no liberties can
        be found, it should only be called once per turn.
        """
        liberties = set()
        for stone in self.stones:
            for liberty in stone.liberties:
                liberties.add(liberty)
        self.liberties = liberties

    def __str__(self):
        """Return a list of the group's stones as a string."""
        return str([str(stone) for stone in self.stones])


class Board(object):
    def __init__(self):
        """Create and initialize an empty board."""
        self.groups: list[Group] = []
        self.current = init.BLACK
        self.influence = np.zeros(
            (19, 19), dtype=int
        )  # 19 * 19 array to hold self influence
        self.stones = np.zeros((19, 19), dtype=int)
        self.record = [self.stones.copy()]
        self.moves = []
    def add(self, point, color):
        is_valid = self.valid_move(point, color)
        if is_valid == 0:
            return None
        self.moves.append(point)
        return self.removed_stones
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
        if self.current == init.BLACK:
            self.current = init.WHITE
            return init.BLACK
        else:
            self.current = init.BLACK
            return init.WHITE
        
    def update_stones(self, trace_back = False):
        if trace_back is True:
            if len(self.record) <= 0: 
                return
            self.record.pop()
            self.stones = self.record[-1].copy()
            print("traceback")
            print(self.record[-1])
            return
        self.stones = np.zeros((19, 19), int)
        for group in self.groups:
            for stone in group.stones:
                if stone.color == init.BLACK:
                    self.stones[stone.point[1]][stone.point[0]] = 1
                elif stone.color == init.WHITE:
                    self.stones[stone.point[1]][stone.point[0]] = -1
        
    def add_record(self):
        # supporting traceback progress
        self.record.append(self.stones.copy())
    
    def ko_handling(self):
        print("in ko handling")
        current_board = self.stones.copy()
        for previous_board in self.record:
            if np.array_equal(current_board ,previous_board):
                print("find unchanged")
                return True
        print("no ko")
        return False

    # * Later
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
                    if self.stones[pos[0]][pos[1]] != 0:
                        continue
                    # print("dx, dy", dx, dy, influence - abs(dx) - abs(dy))
                    temp_influence = bias * max(
                        (influence - abs(dx) - abs(dy)), 0
                    )
                    if self.influence[pos[0]][pos[1]] * bias <= 0:
                        self.influence[pos[0]][pos[1]] += temp_influence
                    else:
                        self.influence[pos[0]][pos[1]] = bias * max(bias * self.influence[pos[0]][pos[1]], bias * temp_influence)

        self.update_stones()

        for group in self.groups:
            for stone in group.stones:
                if stone.color == init.BLACK:
                    propagation(stone.point, 0)
                if stone.color == init.WHITE:
                    propagation(stone.point, 1)
        print(self.influence)
        # print(self.stone)
    
    def valid_move(self, pos, color):
        """
        0: invalid | 1: valid
        """

        # TODO: check overload on other
        # TODO: check self destroy
        # TODO: check ko
        is_present = self.present(pos, color)
        if is_present != 1:
            return is_present 
        
        new_stone = Stone(self, pos, color)
        new_stone.group.update_liberties()
        if len(new_stone.group.liberties) == 0:
            # self destruction
            temp_group = new_stone.group
            if (len(temp_group.stones) > 1):
                new_stone.remove()
                temp_group.update_liberties()
                print(len(temp_group))
                return 0
        
        remove_groups: list[Group] = []
        for group in self.groups:
            if new_stone.group is group:
                continue
            else:
                if len(group.liberties) == 0 and len(group.stones) == 1:
                    remove_groups.append(group)
        for group in remove_groups:
            assert len(group.stones) == 1
            for stone in group.stones:
                spos = stone.point
                self.stones[spos[1]][spos[0]] = 0

        self.stones[pos[1]][pos[0]] = 1 if color == init.BLACK else -1

        if self.ko_handling():
            self.stones = self.record[-1].copy()
            new_stone.remove()
            return 0
        self.update_liberties()
        removed_stones = set()
        for group in self.groups:
            if new_stone.group is group:
                continue
            else:
                if len(group.liberties) == 0:
                    for stone in group.stones:
                        removed_stones.add(stone.point)
                    group.remove()
        self.removed_stones = removed_stones
        print(self.stones)
        self.add_record()
        return 1
    
    def present(self, pos, color):
        if not (0 <= pos[0] < 19 and 0 <= pos[1] < 19):
            return 0
        pos_value = self.stones[pos[1]][pos[0]]
        if pos_value == 0: # EMPTY
            return 1
        return 0
    def update_liberties(self, added_stone : Stone = None):
        """Updates the liberties of the entire board, group by group.

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        # before_len = len(self.groups)
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        # if before_len > len(self.groups):
        #     ko = True
        return True