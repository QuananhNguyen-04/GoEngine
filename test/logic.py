from typing import Union
import numpy as np
import init
import math
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
            if stone.point in liberties:
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
        self.liberties: set[tuple[int, int]] = None
        self.color = stone.color
        self.alive = True

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
    def is_eye_region(self, liberty, visited):
        """
        Checking the surround region of the liberty to determine if it near any enemies
        0: False | 1: Eye
        """
        # neighbors = []
        def bfs(liberty):
            depth = 5
            # neighbors_2 = []
            x, y = liberty
            queue = [(x, y, depth)]
            while queue:
                x, y, cur_depth = queue.pop(0)
                if not visited[y][x]:
                    visited[y][x] = True
                else:
                    continue
                neighbors = []
                if x > 0: 
                    neighbors.append((x - 1, y))
                if y > 0: 
                    neighbors.append((x, y - 1))
                if x < 18: 
                    neighbors.append((x + 1, y))
                if y < 18: 
                    neighbors.append((x, y + 1))
                for pos in neighbors:
                    lx, ly = pos
                    if self.board.stones[ly][lx] == 0:
                        queue.append((ly, lx, cur_depth - 1))
                    else:
                        stone: Stone = self.board.search(point=(lx, ly))
                        if stone.color == self.color:
                            continue
                        else:
                            if not stone.group.alive:
                                continue
                            else:
                                return False
                            
                if cur_depth < 0:
                    break
            return True
        return bfs(liberty)
        
    def is_alive(self):
        if len(self.stones) < 5:
            return False
        if not self.liberties:
            return False
        count = 0
        visited = np.zeros((19, 19), dtype=bool)
        for liberty in self.liberties:
            if visited[liberty[1]][liberty[0]] is True:
                continue
            if self.is_eye_region(liberty, visited) is True:
                count += 1
            if count >= 2:
                self.alive = True
                return True
        self.alive = False
        return False

    def __str__(self):
        """Return a list of the group's stones as a string."""
        return str([str(stone) for stone in self.stones])


class Board(object):
    def __init__(self):
        """Create and initialize an empty board."""
        self.groups: list[Group] = []
        self.current = init.BLACK
        self.grid_size = 19
        self.teritory = []
        self.influence = np.zeros(
            (self.grid_size, self.grid_size), dtype=int
        )  # self.grid_size * self.grid_size array to hold self influence
        self.stones = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.record = [self.stones.copy()]
        self.group_record = [self.groups]
        self.moves = []
        self.captures = {1: 0, -1: 0}
    def add(self, point, color):
        is_valid = self.valid_move(point, color)
        if is_valid == 0:
            return None
        for group in self.groups:
            print(group.liberties)
            group.is_alive()
        self.add_record()
        self.moves.append(point)
        return self.removed_stones
    def search(self, point=None, points=[]):# -> Stone | list[Stone]:
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
            assert len(self.record) == len(self.group_record)
            if len(self.record) <= 0: 
                return
            self.record.pop()
            self.stones = self.record[-1].copy()
            self.group_record.pop()
            self.groups = self.group_record[-1].copy()
            print("traceback")
            # print(self.record[-1])
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
        self.group_record.append(self.groups.copy())
    def ko_handling(self):
        current_board = self.stones.copy()
        for previous_board in self.record:
            if np.array_equal(current_board ,previous_board):
                return True
        return False

    # * Later
    def calculate_influence(self, end=False):
        def propagation(weight, point, color=0):
            if color == 0: # BLACK
                bias = 1
            if color == 1: # WHITE
                bias = -1
            def get_neighbors(x, y):
                neighbors = []
                if x > 0:
                    neighbors.append((x-1, y))
                if y > 0:
                    neighbors.append((x, y-1))
                if x < 18:
                    neighbors.append((x+1, y))
                if y < 18:
                    neighbors.append((x, y+1))
                return neighbors
            x, y = point
            S = 20
            queue: list[tuple[int,int,int]] = [(x, y, weight)]
            visited = []
            while queue:
                cx, cy, rank = queue.pop(0)
                dx = cx - x
                dy = cy - y
                if rank <= 0:
                    break
                if dx == 0 and dy == 0:
                    self.influence[cy][cx] = bias * 130
                else:
                    A = 3 if dx * dy == 0 else 6
                    print(dx, dy)
                    dis = dx * dx + dy * dy
                    print(S / A / dis, self.influence[cy][cx])
                    self.influence[cy][cx] += bias * int(S // A / dis)
                if (cx, cy, rank) in visited:
                    continue
                visited.append((cx, cy, rank))
                for neig in get_neighbors(cx, cy):
                    lx, ly = neig
                    if self.stones[ly][lx] != 0:
                        continue
                    queue.append((lx, ly, rank - 1))
            
        self.update_stones()
        def erosion(erosion_factor = 21):
            for j in range(19):
                for i in range(19):
                    if self.stones[j, i] != 0:
                        continue
                    if self.influence[j, i] > 0:
                        self.influence[j, i] = max(0, self.influence[j, i] - erosion_factor)
                    elif self.influence[j, i] < 0:
                        self.influence[j, i] = min(0, self.influence[j, i] + erosion_factor)

        for group in self.groups:
            weight = 4 if group.alive is True else 4
            for stone in group.stones:
                if stone.color == init.BLACK:
                    propagation(weight, stone.point, 0)
                if stone.color == init.WHITE:
                    propagation(weight, stone.point, 1)
        
        scores = {-1: 0, 1: 0}
        # print(self.influence)
        erosion()
        for row in self.influence:
            for cell in row:
                print((4 - len(str(cell))) * " " + str(cell), end = " ")
                if cell < 0:
                    scores[-1] += 1
                if cell > 0:
                    scores[1] += 1
            print()
        scores[1] -= self.captures[1]
        scores[-1] -= self.captures[-1]
        print("Black", scores[1])
        print("White", scores[-1])
        return scores

        # print(self.stone)
    
    def GNU_algo(self):
        def dilation(board):
            size = board.shape[0]
            new_board = board.copy()
            for i in range(size):
                for j in range(size):
                    if board[i, j] >= 0 and not any(board[adj_i, adj_j] < 0 for adj_i, adj_j in get_adjacent(i, j, size)):
                        new_board[i, j] += sum(1 for adj_i, adj_j in get_adjacent(i, j, size) if board[adj_i, adj_j] > 0)
                    elif board[i, j] <= 0 and not any(board[adj_i, adj_j] > 0 for adj_i, adj_j in get_adjacent(i, j, size)):
                        new_board[i, j] -= sum(1 for adj_i, adj_j in get_adjacent(i, j, size) if board[adj_i, adj_j] < 0)
            return new_board

        def erosion(board):
            size = board.shape[0]
            new_board = board.copy()
            for i in range(size):
                for j in range(size):
                    if board[i, j] > 0:
                        new_board[i, j] -= min(new_board[i, j], sum(1 for adj_i, adj_j in get_adjacent(i, j, size) if board[adj_i, adj_j] <= 0))
                    elif board[i, j] < 0:
                        new_board[i, j] += min(abs(new_board[i, j]), sum(1 for adj_i, adj_j in get_adjacent(i, j, size) if board[adj_i, adj_j] >= 0))
            return new_board

        def get_adjacent(i, j, size):
            adjacent = []
            if i > 0:
                adjacent.append((i-1, j))
            if i < size - 1:
                adjacent.append((i+1, j))
            if j > 0:
                adjacent.append((i, j-1))
            if j < size - 1:
                adjacent.append((i, j+1))
            return adjacent
        def log_board(board):
            for row in board:
                for cell in row:
                    print((4 - len(str(cell))) * " " + str(cell), end = " ")
                    # if cell < 0:
                    #     scores[-1] += 1
                    # if cell > 0:
                    #     scores[1] += 1
                print()
            print(".")
        board = self.stones.copy()
        # Apply 5 dilations
        for _ in range(5):
            board = dilation(board)

        log_board(board)
        # Apply 21 erosions
        for i in range(21):
            board = erosion(board)
            if i % 6 == 1:
                print(i)
                log_board(board)
        log_board(board)
    
    def get_valid_moves(self, color):
        self.valid_moves = []
        # print(self.stones)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.valid_move((j, i), color, True) == 1:
                    self.stones = self.record[-1].copy()
                    self.groups = self.group_record[-1].copy()
                    self.valid_moves.append((j, i))
                # print(len(self.record), len(self.group_record))
        # print(self.stones)
        return self.valid_moves
    def valid_move(self, pos, color, temp = False):
        """
        0: invalid | 1: valid
        """

        # TODO: check overload on other
        # TODO: check self destroy
        # TODO: check ko
        is_present = self.present(pos)
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
        if temp is False:
            self.update_liberties()
            removed_stones = set()
            for group in self.groups:
                color = 1 if group.color == init.BLACK else -1 if group.color == init.WHITE else 0
                if new_stone.group is group:
                    continue
                else:
                    if len(group.liberties) == 0:
                        for stone in group.stones:
                            removed_stones.add(stone.point)
                            self.captures[color] += 1
                        group.remove()
            self.removed_stones = removed_stones
        # print(self.stones)
        # print(len(self.groups))
        return 1
    
    def present(self, pos):
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

    def get_neighbors(self, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x-1, y))
        if y > 0:
            neighbors.append((x, y-1))
        if x < 18:
            neighbors.append((x+1, y))
        if y < 18:
            neighbors.append((x, y+1))
        return neighbors
    
    def calculate_scores(self):
        stones_visited = np.zeros((self.grid_size, self.grid_size), dtype=int)
        teritory = np.zeros((self.grid_size, self.grid_size), dtype=int)
        class Stone_Type:
            BLACK = 1
            WHITE = -1
            EMPTY = 0

        scores = {Stone_Type.BLACK: 0, Stone_Type.WHITE : 0}

        def BFS(y, x):
            queue = [(x, y)]
            land_list = []
            owner = None
            while queue:
                x0, y0 = queue.pop(0)
                if stones_visited[y0, x0] == 1:
                    continue
                land_list.append((x0, y0))
                stones_visited[y0, x0] = 1
                neighbors = self.get_neighbors(x0, y0)
                for (lx, ly) in neighbors:
                    
                    if self.stones[ly, lx] != Stone_Type.EMPTY:
                        stone = self.search((lx, ly))
                        if stone.group.alive is False:
                            continue
                        if owner is None:
                            owner = self.stones[ly, lx]
                        else:
                            if owner != self.stones[ly, lx]:
                                owner = 0
                        continue
                    queue.append((lx, ly))
            if owner == 0 or owner is None:
                return land_list
            for (x, y) in land_list:
                teritory[y, x] = owner
            return land_list

        def land_region():
            self.teritory.clear()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.stones[i, j] == Stone_Type.EMPTY and stones_visited[i, j] == 0:
                        teri = BFS(i, j)
                        if teri is not None:
                            self.teritory.append(teri)
        new_groups = self.connected_group()
        land_region()
        print("territory", len(self.teritory))
        for region in self.teritory:
            print(len(region))
        print(self.captures[1], self.captures[-1])
        print(teritory)
        print("Black", scores[Stone_Type.BLACK] - self.captures[1])
        print("White", scores[Stone_Type.WHITE] - self.captures[-1])
    def connected_group(self):
        board_stone_visited = np.zeros((self.grid_size, self.grid_size), dtype=int)
        def potential_connection(x, y, color):
            stone_visited = np.zeros((self.grid_size, self.grid_size), dtype=int)
            queue = [(x, y)]
            group = []
            board_stone_visited[y, x] = 1
            while queue:
                is_stone = False
                cx, cy = queue.pop()
                if stone_visited[cy, cx] == 1:
                    continue
                stone_visited[cy, cx] = 1
                if self.stones[cy, cx] == color:
                    group.append((cx, cy))
                    is_stone = True
                neighbors = self.get_neighbors(cx, cy)
                for (nx, ny) in neighbors:
                    if self.stones[ny, nx] == color:
                        board_stone_visited[ny, nx] = 1
                        queue.append((nx, ny))
                    if self.stones[ny, nx] == 0 and is_stone is True:
                        queue.append((nx, ny))
            return group

        connected_groups = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.stones[i, j] != 0 and board_stone_visited[i, j] == 0:
                    connected_groups.append(potential_connection(j, i, self.stones[i, j]))
        print("nums of groups", len(connected_groups), connected_groups)
        print(board_stone_visited)
        return connected_groups
