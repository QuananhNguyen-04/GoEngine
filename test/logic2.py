import numpy as np
import init


class Board:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.stones = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.record = [self.stones.copy()]
        self.captures = {1: 0, -1: 0}
        self.current = init.BLACK
        self.removed_stones = []
        self.moves = []

    def get_neighbors(self, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if x < self.grid_size - 1:
            neighbors.append((x + 1, y))
        if y < self.grid_size - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def turn(self):
        """Keep track of the turn by flipping between BLACK and WHITE."""
        if self.current == init.BLACK:
            self.current = init.WHITE
            return init.BLACK
        else:
            self.current = init.BLACK
            return init.WHITE

    def undo(self):
        if len(self.record) > 1:
            self.record.pop()
            self.stones = self.record[-1].copy()

    def get_groups(self, point, color):
        stone_visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = [point]
        group = []
        liberties = []
        while queue:
            cx, cy = queue.pop(0)
            if stone_visited[cy, cx]:
                continue
            stone_visited[cy, cx] = True
            if self.stones[cy, cx] == color:
                group.append((cx, cy))
                neighbors = self.get_neighbors(cx, cy)
                for nx, ny in neighbors:
                    if self.stones[ny, nx] == color:
                        queue.append((nx, ny))
                    if self.stones[ny, nx] == 0:
                        liberties.append((nx, ny))
        return group, liberties

    def check_ko(self):
        for record in self.record:
            if np.array_equal(record, self.stones):
                return True
        return False

    def check_atari(self, point, color, captures=True):
        """
        check if move is atari
        param:
            `point`: (x, y) stone coordinates (0 - 18)
            `color`: stone color 1 or -1
        return:
            False if move is ko case else True
        """
        x, y = point
        neighbors = self.get_neighbors(x, y)
        for nx, ny in neighbors:
            if self.stones[ny, nx] != color:
                pre_color = self.stones[ny, nx]
                if pre_color == 0:
                    continue
                group, liberties = self.get_groups((nx, ny), pre_color)
                if len(liberties) == 0:
                    if len(group) == 1:
                        cx, cy = group[0]
                        self.stones[cy, cx] = 0
                        if self.check_ko():
                            print("ko case")
                            self.stones[cy, cx] = pre_color
                            return False
                        else:
                            self.remove(group, pre_color, captures)
                            # return True
                    else:
                        self.remove(group, color, captures)
                        # return True
        # print("not atari")
        return not self.check_self_atari(point, color)

    def check_self_atari(self, point, color):
        _, self_liberties = self.get_groups(point, color)
        # print("self_liberties: ", self_liberties)
        if len(self_liberties) == 0:
            return True
        return False

    def remove(self, group, color, capture=False):

        for x, y in group:
            self.removed_stones.append((x, y))
            self.stones[y, x] = 0

        if capture is True:
            self.captures[color] += len(group)

    def is_duplicated(self, point):
        x, y = point
        if self.stones[y, x] != 0:
            return False
        return True

    def is_valid_move(self, point, color):
        x, y = point
        # print("check duplicated")
        if self.is_duplicated(point) is False:
            return False
        self.stones[y, x] = color
        # print("check atari")
        # check atari
        if self.check_atari(point, color) is False:
            return False
        # print("valid")
        return True

    def get_stone_type(self, color):
        stone_type = 1 if color == init.BLACK else -1 if color == init.WHITE else 0
        return stone_type
    
    def add(self, point, color):
        self.removed_stones = []
        # check if move is ko
        # check if move is self atari
        stone_type = self.get_stone_type(color)

        if self.is_valid_move(point, stone_type) is False:
            self.stones = self.record[-1].copy()
            return False

        self.record.append(self.stones.copy())
        # return False
        return self.removed_stones

    def get_valid_moves(self, color):
        stone_type = self.get_stone_type(color)
        valid_moves = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.is_valid_move((j, i), stone_type) is True:
                    valid_moves.append((j, i))
                # else: 
                #     print("invalid move", (j, i))
        return valid_moves

    def get_all_groups(self):
        visited = np.zeros_like(self.stones, dtype=bool)
        groups: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if visited[i, j] is False and self.stones[i, j] != 0:
                    group, liberties = self.get_groups((j, i), self.stones[i, j])
                    groups.append((group, liberties))
                    for x, y in group:
                        visited[y, x] = True
        return groups

    def dilation_influence(self, influence_board):
        def empty_dilations(x, y):
            neighbors = self.get_neighbors(x, y)
            near_white = 0
            near_black = 0 
            for nx, ny in neighbors:
                if influence_board[ny, nx] == 0:
                    continue
                if influence_board[ny, nx] < 0:
                    near_white += 1
                else:
                    near_black += 1
            if near_white > 0 and near_black > 0:
                return
            if near_black == 0 and near_white == 0:
                return
            if near_white > 0:
                temp_influence[y, x] -= near_white
            if near_black > 0:
                temp_influence[y, x] += near_black
            # print("influence", influence_board[y, x])
        def stone_dilations(x, y, color):
            neighbors = self.get_neighbors(x, y)
            near_opponent = 0
            empty = 0
            bias = 1 if color == 1 else -1
            for nx, ny in neighbors:
                if influence_board[ny, nx] == 0:
                    empty += 1
                    continue
                if influence_board[ny, nx] * bias < 0:
                    near_opponent += 1
            if near_opponent > 0:
                return
            temp_influence[y, x] += bias * (len(neighbors) - empty)

        # groups = self.get_all_groups()
        """ #TODO: 
            working on once influence, currently the influence is updated for each stone dilation 
          resulting influence area by empty stone.
        """
        temp_influence = np.zeros_like(influence_board, dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.stones[i, j] == 0:
                    empty_dilations(j, i)
                else:
                    stone_dilations(j, i, self.stones[i, j])                
        influence_board += temp_influence
        # self.log_board(influence_board)

    def erosion_influence(self, influence_board):
        def stone_erosion(x, y, color):
            neighbors = self.get_neighbors(x, y)
            near_opponent = 0
            bias = 1 if color > 0 else -1
            for nx, ny in neighbors:
                if influence_board[ny, nx] * bias <= 0:
                    near_opponent += 1
            
            # print("near opponent",x ,y, near_opponent, bias)
            temp_influence[y, x] += bias * near_opponent

        temp_influence = np.zeros_like(influence_board, dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if influence_board[i, j] != 0:
                    stone_erosion(j, i, influence_board[i, j])
        # self.log_board(temp_influence)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                a = temp_influence[i, j]
                b = influence_board[i, j]
                if (b - a) * b < 0:
                    influence_board[i, j] = 0
                else:
                    influence_board[i, j] -= temp_influence[i, j]
        # self.log_board(influence_board)
    def groups_influence(self):
        influence_board = self.stones.copy() * 128
        n, m = 5, 11
        for _ in range(n):
            self.dilation_influence(influence_board)
        self.log_board(influence_board)
        for _ in range(m):
            self.erosion_influence(influence_board)
        self.log_board(influence_board)
        score = {1: 0, -1: 0}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if influence_board[i, j] > 0:
                    score[1] += 1
                elif influence_board[i, j] < 0:
                    score[-1] += 1
        print("Black: ", score[1], "White: ", score[-1])
    def log_board(self,board, infile = None):
        print('[', file=infile)
        for row in board:
            print('[',end='', file=infile)
            for idx, cell in enumerate(row):
                print((2 - len(str(cell))) * " " + str(cell), end="", file=infile)
                # if cell < 0:
                #     scores[-1] += 1
                # if cell > 0:
                #     scores[1] += 1
                if idx < len(row) - 1:
                    print(',',end=' ', file=infile)
            print('],', file=infile)
        print("],", file=infile)