import sgf
import random

class Agent:
    def __init__(self) -> None:

        self.moves = []
        # self.moves = [
        #     [16, 3],
        #     [3, 15],
        #     [15, 16],
        #     [3, 3],
        #     [2, 2],
        #     [3, 2],
        #     [2, 3],
        #     [3, 4],
        #     [1, 5],
        #     [14, 2],
        #     [15, 4],
        #     [16, 14],
        #     [13, 15],
        #     [15, 10],
        #     [11, 2],
        #     [13, 3],
        #     [8, 2],
        #     [16, 6],
        #     [14, 6],
        #     [12, 4],
        #     [15, 2],
        #     [16, 8],
        #     [2, 13],
        #     [5, 16],
        #     [2, 6],
        #     [6, 2],
        #     [7, 2],
        #     [6, 3],
        #     [8, 4],
        #     [10, 4],
        #     [4, 1],
        #     [7, 4],
        #     [8, 5],
        #     [9, 3],
        #     [6, 1],
        #     [5, 1],
        #     [5, 2],
        #     [8, 3],
        #     [7, 3],
        #     [6, 4],
        #     [9, 2],
        #     [10, 2],
        #     [10, 1],
        #     [10, 3],
        #     [8, 1],
        #     [6, 6],
        #     [8, 7],
        #     [10, 7],
        #     [6, 8],
        #     [3, 7],
        #     [3, 6],
        #     [4, 6],
        #     [10, 8],
        #     [11, 8],
        #     [10, 9],
        #     [11, 9],
        #     [5, 3],
        #     [5, 4],
        #     [9, 7],
        #     [10, 6],
        #     [10, 10],
        #     [5, 8],
        #     [6, 9],
        #     [2, 7],
        #     [1, 7],
        #     [5, 9],
        #     [6, 10],
        #     [2, 14],
        #     [4, 5],
        #     [5, 7],
        #     [5, 5],
        #     [3, 5],
        #     [6, 5],
        #     [7, 5],
        #     [4, 4],
        #     [7, 6],
        #     [4, 3],
        #     [3, 13],
        #     [11, 10],
        #     [8, 16],
        #     [9, 5],
        #     [15, 5],
        #     [3, 12],
        #     [2, 12],
        #     [2, 11],
        #     [1, 13],
        #     [4, 10],
        #     [2, 10],
        #     [2, 8],
        #     [3, 10],
        #     [3, 9],
        # ]
        self.next = 0
        self.loaded = False

    def convert_sgf_to_pos(self, move):
        SGF_POS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        x = SGF_POS.index(move[0][0]) + 1
        y = SGF_POS.index(move[0][1]) + 1
        return (x, y)

    def read_sgf(self, src):
        with open(src, "r", encoding="utf-8") as file:
            sgf_content = file.read()

        # Parse the SGF content
        collection = sgf.parse(sgf_content)

        # Access the first game record
        game = collection[0]
        root = game.root
        print(f"  Black player: {root.properties.get('PB')}")
        print(f"  White player: {root.properties.get('PW')}")
        print(f"  Result: {root.properties.get('RE')}")
        current_node = game.rest
        if current_node:
            for node in current_node:
                move = node.properties.get("B") or node.properties.get("W")
                # if move:
                #     player = 'Black' if 'B' in node.properties else 'White'
                #     print(f"{player} move: {move}")
                self.moves.append(self.convert_sgf_to_pos(move))
        self.loaded = True
        return root.properties.get('RE')

    def play(self):
        self.next += 1
        if self.next == len(self.moves):
            self.loaded = False
        return self.moves[self.next - 1]

    def auto_play(self, valid_moves):
        random.shuffle(valid_moves)
        choice = random.choices(valid_moves, k=5)

        pass

class Evaluation:
    def __init__(self):
        
        pass