import pygame
import UI
import logic
import init 

class Game():
    def __init__(self):
        pygame.display.set_caption("Go")
        pass
    def run(self):
        screen = pygame.display.set_mode(init.BOARD, 0, 32)
        background = pygame.image.load(init.BACKGROUND).convert()
        ui_board = UI.UIBoard(screen, background)
        board = logic.Board()
        running = True
        # border = board.outline
        border = pygame.Rect(ui_board.outline[0] - ui_board.square_size // 2, ui_board.outline[1] - ui_board.square_size // 2, ui_board.square_size * 19 + ui_board.square_size // 2, ui_board.square_size * 19 + ui_board.square_size // 2)
        while running:
            # pygame.time.wait(250)
            event = pygame.event.wait()
            # for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1 and border.collidepoint(event.pos):
                    # use round to get the nearest integer
                    x = round ((event.pos[0] - ui_board.outline[0]) / ui_board.square_size)
                    y = round ((event.pos[1] - ui_board.outline[1]) / ui_board.square_size)
                    print("stone coord",x, y)
                    result = board.add((x, y), board.current)
                    if result is None:
                        print("None")
                    else: 
                        ui_board.remove(result)
                        if hasattr(result, "__iter__") or len(result) == 0:
                            print("success")
                            UI.UIStone(ui_board, (x, y), board.turn())


            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # board.undo()
                    pass
                elif event.key == pygame.K_p:
                    ui_board.turn()
                elif event.key == pygame.K_r:
                    running = False
                    return True
                elif event.key == pygame.K_SPACE:
                    # calculate and end the game
                    ui_board.calculate_winner()
        return False
    
    def start(self):
        if self.run() is True:
            self.start()
            return