import sgf

# Load SGF file
with open('./3ff4-gokifu-20240715-Lian_Xiao-Fan_Tingyu.sgf', 'r', encoding="utf-8") as file:
    sgf_content = file.read()

# Parse the SGF content
collection = sgf.parse(sgf_content)

# Access the first game record
game = collection[0]

# Get the root node
root : sgf.Node = game.root

# Print some properties
print("Game info:")
print(f"  Game: {root.properties.get('GM')}")
print(f"  Black player: {root.properties.get('PB')}")
print(f"  White player: {root.properties.get('PW')}")
print(f"  Result: {root.properties.get('RE')}")

# Iterate through the game tree to get the moves
current_node = game.rest
if current_node:
    for node in current_node:
        move = node.properties.get('B') or node.properties.get('W')
        # if move:
        #     player = 'Black' if 'B' in node.properties else 'White'
        #     print(f"{player} move: {move}")
        print(type(move[0]), move[0])