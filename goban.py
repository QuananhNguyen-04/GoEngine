#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import pygame
import go
# from sys import exit

BACKGROUND = "images/ramin.jpg"
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Stone(go.Stone):
    def __init__(self, board, point, color):
        """Create, initialize and draw a stone."""
        self.square_size = 32
        self.stone_size = self.square_size // 2
        super(Stone, self).__init__(board, point, color)
        self.coords = (board.outline[0] + self.point[0] * self.square_size, board.outline[1] + self.point[1] * self.square_size)
        print(self.coords)
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(screen, self.color, self.coords, self.square_size // 2, 0)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (self.coords[0] - self.stone_size, self.coords[1] - self.stone_size)
        area_rect = pygame.Rect(blit_coords, (self.square_size, self.square_size))
        screen.blit(background, blit_coords, area_rect)
        pygame.display.update()
        super(Stone, self).remove()


class Board(go.Board):
    def __init__(self):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__()
        self.square_size = 32
        self.grid_size = 19
        self.outline = pygame.Rect(
            BOARD_SIZE[0] // 2 - self.grid_size // 2 * self.square_size,
            BOARD_SIZE[1] // 2 - self.grid_size // 2 * self.square_size,
            (self.grid_size - 1) * self.square_size,
            (self.grid_size - 1) * self.square_size,
        )
        print(self.outline)
        self.draw()

    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        # self.outline.inflate_ip(15, 15)
        # grid drawing
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(
                    self.outline[0] + (self.square_size * i),
                    self.outline[1] + (self.square_size * j),
                    self.square_size,
                    self.square_size,
                )
                pygame.draw.rect(background, BLACK, rect, 1)
        # 9 stars point drawing
        for i in range(3):
            for j in range(3):
                coords = (
                    self.outline[0] + 3 * self.square_size + (6 * self.square_size * i),
                    self.outline[1] + 3 * self.square_size + (6 * self.square_size * j),
                )
                pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        if added_stone:
            added_stone.group.update_liberties()


def main():
    running = True
    # border = board.outline
    border = pygame.Rect(board.outline[0] - board.square_size // 2, board.outline[1] - board.square_size // 2, board.square_size * 19 + board.square_size // 2, board.square_size * 19 + board.square_size // 2)
    while running:
        # pygame.time.wait(250)
        event = pygame.event.wait()
        # for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1 and border.collidepoint(event.pos):
                # use round to get the nearest integer
                x = round ((event.pos[0] - board.outline[0]) / board.square_size)
                y = round ((event.pos[1] - board.outline[1]) / board.square_size)
                print("stone coord",x, y)
                stone = board.search(point=(x, y))
                if stone:
                    stone.remove()
                else:
                    added_stone = Stone(board, (x, y), board.turn())
                board.update_liberties(added_stone)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_z:
                # board.undo()
                pass
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Goban")
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    board = Board()
    main()
