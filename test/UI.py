import pygame
import numpy as np
import init

# from sys import exit

BACKGROUND = "images/ramin.jpg"
BOARD_SIZE = init.BOARD
WHITE = init.WHITE
BLACK = init.BLACK


class UIBoard:
    def __init__(self, screen, background):
        """Create, initialize and draw an empty board."""
        super().__init__()
        self.screen = screen
        self.background = background
        self.square_size = 32
        self.grid_size = 19
        self.outline = pygame.Rect(
            BOARD_SIZE[0] // 2 - self.grid_size // 2 * self.square_size,
            BOARD_SIZE[1] // 2 - self.grid_size // 2 * self.square_size,
            (self.grid_size - 1) * self.square_size,
            (self.grid_size - 1) * self.square_size,
        )
        print(self.outline)
        self.stones = []
        self.temp_stones = None
        self.draw()

    def remove(self, stones):
        if hasattr(stones, "__iter__"):
            # print(stones)
            temp_list = self.stones.copy()
            for stone in temp_list:
                if stone.point in stones:
                    stone.remove()

    def undo(self, record):
        print("undo")
        # print(record)
        # print(len(self.stones))
        temp_list = self.stones.copy()
        for stone in temp_list:
            # print(stone.point, sep=" ")
            stone.remove()

        # print(len(self.stones))
        for y, row in enumerate(record):
            for x, stone in enumerate(row):
                # print(x, y, stone)
                if stone == 0:
                    continue
                stone = UIStone(self, (x, y), init.BLACK if stone == 1 else init.WHITE)
        # print(len(self.stones))

    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(self.background, BLACK, self.outline, 3)
        # grid drawing
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(
                    self.outline[0] + (self.square_size * i),
                    self.outline[1] + (self.square_size * j),
                    self.square_size,
                    self.square_size,
                )
                pygame.draw.rect(self.background, BLACK, rect, 1)
        # 9 stars point drawing
        for i in range(3):
            for j in range(3):
                coords = (
                    self.outline[0] + 3 * self.square_size + (6 * self.square_size * i),
                    self.outline[1] + 3 * self.square_size + (6 * self.square_size * j),
                )
                pygame.draw.circle(self.background, BLACK, coords, 5, 0)
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

    def add_temp(self, point, color, temp=True):
        for stone in self.stones:
            if stone.point == point:
                return
        if self.temp_stones is None:
            if point[0] < 0 or point[1] < 0 or point[0] > 18 or point[1] > 18:
                return
            new_color = (color[0], color[1], color[2], 255)
            self.temp_stones = UIStone(self, point, new_color, True)
        else:
            if point[0] < 0 or point[1] < 0 or point[0] > 18 or point[1] > 18:
                self.temp_stones.remove()
                self.temp_stones = None
                return
            if point == self.temp_stones.point:
                return

            for stone in self.stones:
                if stone.point == self.temp_stones.point:
                    self.temp_stones.remove()
                    self.temp_stones = None
                    stone.draw()
                    return
            self.temp_stones.remove()
            new_color = (color[0], color[1], color[2], 255)
            self.temp_stones = UIStone(self, point, new_color, True)


class UIStone:
    def __init__(self, board: UIBoard, point, color, sub_board=False):
        """Create, initialize and draw a stone."""
        self.board = board
        self.square_size = 32
        self.stone_size = self.square_size // 2
        self.color = color
        self.point = point
        self.coords = (
            board.outline[0] + self.point[0] * self.square_size,
            board.outline[1] + self.point[1] * self.square_size,
        )
        self.draw()
        if not sub_board:
            self.board.stones.append(self)

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(
            self.board.screen, self.color, 
            self.coords, self.square_size // 2, 0
        )
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (
            self.coords[0] - self.stone_size,
            self.coords[1] - self.stone_size,
        )
        area_rect = pygame.Rect(blit_coords, (self.square_size, self.square_size))
        self.board.screen.blit(self.board.background, blit_coords, area_rect)
        pygame.display.update()

        if self in self.board.stones:
            self.board.stones.remove(self)
        del self
