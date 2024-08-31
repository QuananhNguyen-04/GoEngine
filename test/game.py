import numpy as np
import pygame
import random
import UI
import logic2
import init
import agent
import os

random.seed(42)
# print("[", end="", file=open("y_input.txt", "w", encoding="utf-8"))
# print("[", file=open("X_input.txt", "w", encoding="utf-8"))
class Game:
    def __init__(self):
        pygame.display.set_caption("Go")
        self.record = []
        self.record_result = []
        self.moves = []

    def run(self, kifu=None):
        screen = pygame.display.set_mode(init.BOARD, 0, 32)
        background = pygame.image.load(init.BACKGROUND).convert()
        ui_board = UI.UIBoard(screen, background)

        player = agent.Agent()
        # evaluation = agent.Evaluation()

        game_file = kifu if kifu is not None else "./sgf_files/3f8w-gokifu-20240529-Li_Weiqing-Ke_Jie.sgf"
        result = player.read_sgf(game_file)
        winner = 0
        if result[0][0] == "B":
            winner = 1
        elif result[0][0] == "W":
            winner = -1

        def is_result_alpha(char: str):
            return char.isalpha()

        if result[0][2:] == "R" or is_result_alpha(result[0][2]):
            winner *= 15
        else:
            winner *= float(result[0][2:])

        board = logic2.Board(19)
        running = True
        moves = []
        def agent_step():
            if player.loaded:
                x, y = player.play()
                x -= 2
                y -= 2
                # print("stone coord", x, y)

                result = board.add((x, y), board.current)
                if result is False:
                    print("None")
                else:
                    moves.append((x, y))
                    board.turn()
                    # ui_board.remove(result)
                    # if hasattr(result, "__iter__") or len(result) == 0:
                    #     # print("success")
                    #     # print(result)
                    #     UI.UIStone(ui_board, (x, y), board.turn())
                return True
            return False

        border = pygame.Rect(
            ui_board.outline[0] - ui_board.square_size // 2,
            ui_board.outline[1] - ui_board.square_size // 2,
            ui_board.square_size * 19 + ui_board.square_size // 2,
            ui_board.square_size * 19 + ui_board.square_size // 2,
        )
        while running:
            # pygame.time.wait(250)
            # for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP] or kifu is not None:
                while agent_step():
                    # pygame.time.delay(2)
                    pass
                # if agent_step() is False: 
                #     pass
                running = False
                continue
            
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                x = round((mouse_pos[0] - ui_board.outline[0]) / ui_board.square_size)
                y = round((mouse_pos[1] - ui_board.outline[1]) / ui_board.square_size)
                ui_board.add_temp((x, y), board.current)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and border.collidepoint(event.pos):
                x = round((event.pos[0] - ui_board.outline[0]) / ui_board.square_size)
                y = round((event.pos[1] - ui_board.outline[1]) / ui_board.square_size)
                result = board.add((x, y), board.current)
                ui_board.remove(result)
                if result is False:
                    print("None")

                elif hasattr(result, "__iter__") or len(result) == 0:
                    # print("success")
                    # print(result)
                    UI.UIStone(ui_board, (x, y), board.turn())

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif (
                    event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    print(len(board.record))
                    if len(board.record) > 1:
                        board.undo()
                        ui_board.undo(board.record[-1])
                        board.turn()

                elif event.key == pygame.K_SLASH:
                    print(len(board.get_valid_moves(board.current)))
                    print("done")

                elif event.key == pygame.K_p:
                    board.turn()

                elif event.key == pygame.K_r:
                    running = False
                    return True
                elif event.key == pygame.K_SPACE:
                # calculate and end the game
                    board.groups_influence()

                elif event.key == pygame.K_l or event.key == pygame.K_RIGHT:
                    agent_step()
                    
                elif event.key == pygame.K_e:
                    print("evaluate")
                    # evaluation.evaluate(board.record)
                    # print(winner)
                # print(board.stones)
        # print("[", file=open("X_input.txt", "a", encoding="utf-8"))
        # for state in board.record[-3:]:
            # board.log_board(state, open("X_input.txt", "a", encoding="utf-8"))
        # print("],", file=open("X_input.txt", "a", encoding="utf-8"))
        # print("retrain")
        board.record.insert(0, np.zeros_like(board.stones))
        board.record.insert(0, np.zeros_like(board.stones))
        self.record.append(board.record) # shape (1, len + 2, 19, 19)
        self.record_result.append(winner) # shape (1, 1)
        self.moves.append(moves)
        return False

    def start(self):
        pygame.init()
        sgf_files = [
            "./sgf_files/" + file
            for file in os.listdir("./sgf_files")
            if file.endswith(".sgf")
        ]
        random.shuffle(sgf_files)
        for idx, kifu_file in enumerate(sgf_files[:100]):
            print(f"Processing kifu no. {idx + 1}")
            self.run(kifu_file)
        pygame.quit()
        evaluation = agent.Evaluation()
        evaluation.train(self.record, self.record_result)
        # evaluation.cross_validation(self.record, self.record_result)

        # decision = agent.Decision()
        # decision.train(self.record, self.moves)
        # print("]", file=open("X_input.txt", "a", encoding="utf-8"))
        # print("]", file=open("y_input.txt", "a", encoding="utf-8"))
