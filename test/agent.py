import sgf


class Agent():
    def __init__(self) -> None:
        self.moves = []
        self.next = 0
        self.loaded = False
    def convert_sgf_to_pos(self, move):
        SGF_POS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        x = SGF_POS.index(move[0][0]) + 1
        y = SGF_POS.index(move[0][1]) + 1
        return (x, y)
    def read_sgf(self, src):
        with open(src,'r', encoding="utf-8") as file:
            sgf_content = file.read()

        # Parse the SGF content
        collection = sgf.parse(sgf_content)

        # Access the first game record
        game = collection[0]
        current_node = game.rest
        if current_node:
            for node in current_node:
                move = node.properties.get('B') or node.properties.get('W')
                # if move:
                #     player = 'Black' if 'B' in node.properties else 'White'
                #     print(f"{player} move: {move}")
                self.moves.append(self.convert_sgf_to_pos(move))
        self.loaded = True
    def play(self):
        self.next += 1
        if self.next == len(self.moves):
            self.loaded = False
        return self.moves[self.next - 1]
    