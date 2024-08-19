import sgf
import random
import os
from sklearn.model_selection import train_test_split
import torch
# import intel_extension_for_pytorch as ipex
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from AI import AI, Rollout

torch.manual_seed(42)
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
        return root.properties.get("RE")

    def play(self):
        self.next += 1
        if self.next == len(self.moves):
            self.loaded = False
        return self.moves[self.next - 1]

    def auto_play(self, valid_moves):
        # random.shuffle(valid_moves)
        # choice = random.choices(valid_moves, k=5)

        pass


class Evaluation:
    def __init__(self):
        self.model = AI()
        if os.path.isfile("model1.pth"):
            self.model.load_state_dict(torch.load("model1.pth", weights_only=True))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)

    def evaluate(self, record):
        state = []
        model = self.model
        print(type(model))
        if len(record) < 3:
            for _ in range(3 - len(record)):
                state.append(np.zeros((19, 19), np.float32))
        for move in record[-3:]:
            state.append(move)
        assert len(state) == 3
        X = torch.from_numpy(np.array([state], dtype=np.float32))
        assert X.shape == (1, 3, 19, 19)
        model.eval()
        with torch.inference_mode():
            y_pred = model(X)
            # test_loss = self.loss_fn(y_pred, y_test)
            print(y_pred)

        pass

    def transfer_data(self, record: list, result: list):
        new_record: list = record.copy()
        # new_record.insert(0, np.zeros((19, 19), np.float32))
        # new_record.insert(0, np.zeros((19, 19), np.float32))
        state_answer = []
        state = []
        # print(len(record), len(result))
        for mini_record, mini_result in zip(new_record, result):
            # mini_record.shape = (game_depth + 2, 19, 19)
            # print(len(mini_record), mini_result)
            game_depth = len(mini_record)
            for idx, (state0, state1, state2) in enumerate(
                zip(mini_record, mini_record[1:], mini_record[2:])
            ):
                # print("reading mini_record", state0.shape, state1.shape, state2.shape)
                mini_state = []
                mini_state.append(state0)
                mini_state.append(state1)
                mini_state.append(state2)
                state.append(mini_state)
                if idx == game_depth - 3:
                    local_X = torch.from_numpy(np.array([mini_state], dtype=np.float32))
                    # print(local_X.shape)
                    assert local_X.shape == (1, 3, 19, 19)
                    # pred_score = model(local_X)
                    # print(pred_score.item(), mini_result)
                score = (
                    mini_result * (idx) - 6.5 * (game_depth - idx - 1)
                ) / game_depth + random.randint(-5, 5) / random.randint(1, 5) / 2

                state_answer.append([score])
        # print(len(record))
        X = torch.from_numpy(np.array(state, dtype=np.float32))
        y = torch.from_numpy(np.array(state_answer, dtype=np.float32))

        return X, y

    def retrain(self, record: list[torch.Tensor], result: list[torch.Tensor]):
        def split_data(X, y, test_size=50):
            test_size = (len(X) // 4) if len(X) < 200 else len(X) // 3 + test_size
            randp = torch.randperm(len(X))
            X_test = X[randp[:test_size]]
            y_test = y[randp[:test_size]]
            X_train = X[randp[test_size:]]
            y_train = y[randp[test_size:]]
            return X_train, y_train, X_test, y_test

        model = self.model
        X, y = self.transfer_data(record, result)
        X_train = X[(len(X) // 8) :]
        y_train = y[(len(y) // 8) :]
        X_test = X[: (len(X) // 8)]
        y_test = y[: (len(y) // 8)]

        # X_train, y_train, X_test, y_test = split_data(X, y, 200)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        # assert X.shape == (len(record), 3, 19, 19)
        # assert y.shape == (len(record), 1)
        data_loader = torch.utils.data.DataLoader(
            TensorDataset(X_train, y_train), batch_size=64, shuffle=True
        )
        epochs = 50
        for epoch in range(epochs):
            loss_per_epoch = 0.0
            model.train()
            with tqdm(
                data_loader, unit="step", ncols=80, desc=f"Epoch {epoch + 1}/{epochs}"
            ) as tepoch:
                for batch_X, batch_y in tepoch:
                    y_pred = model(batch_X)
                    loss: torch.Tensor = self.loss_fn(y_pred, batch_y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_per_epoch += loss.item()
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            if (epoch + 1) % 5 == 0:
                with torch.inference_mode():
                    y_pred: torch.Tensor = model(X_test)
                    # print(y_pred.shape, y_test.shape)
                    test_loss = self.loss_fn(y_pred, y_test)
                    print(
                        f"Epoch: {epoch + 1} | Loss : {loss_per_epoch/len(data_loader):.4f} | Test Loss: {test_loss.item():.4f}"
                    )

        model.eval()
        with torch.inference_mode():
            y_pred: torch.Tensor = model(X_test)
            # print(y_pred.shape, y_test.shape)
            test_loss = self.loss_fn(y_pred, y_test)
            print(
                f"Epoch: {epoch + 1} | Loss : {loss_per_epoch/len(data_loader):.4f} | Test Loss: {test_loss.item():.4f}"
            )

        model.eval()
        with torch.inference_mode():
            y_pred = model(X_test)
            test_loss = self.loss_fn(y_pred, y_test)
            for pred, actual in zip(y_pred[20:30], y_test[20:30]):
                print(pred.item(), actual.item())
            print(test_loss.item())

        # torch.save(model.state_dict(), "model2.pth")

    def cross_validation(self, record, result):
        X, y = self.transfer_data(record, result)

        epochs = 20
        num_folds = 5
        total_samples = X.size(0)
        part = total_samples // num_folds

        # Create folds
        folds = [
            (X[i * part : (i + 1) * part], y[i * part : (i + 1) * part])
            for i in range(num_folds)
        ]
        # Handle any remaining samples for the last fold
        if total_samples % num_folds != 0:
            folds[-1] = (
                torch.cat([folds[-1][0], X[part * num_folds :]]),
                torch.cat([folds[-1][1], y[part * num_folds :]]),
            )
        for i in range(num_folds):
            # Define the validation set and training set
            X_val, y_val = folds[i]

            # Combine other folds for training
            X_train = torch.cat([folds[j][0] for j in range(num_folds) if j != i])
            y_train = torch.cat([folds[j][1] for j in range(num_folds) if j != i])

            # Create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            # val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Define your model, loss function, and optimizer

            self.model.load_state_dict(torch.load("model1.pth", weights_only=True))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

            # Training Loop
            self.model.train()
            for epoch in range(epochs):
                # loss_per_epoch = 0.0
                with tqdm(
                    train_loader, unit="step", ncols=80, desc=f"Epoch {epoch + 1}/{epochs}" 
                ) as tepoch:
                    for batch_X, batch_y in tepoch:
                        self.optimizer.zero_grad()
                        y_pred = self.model(batch_X)
                        loss: torch.Tensor = self.loss_fn(y_pred, batch_y)
                        loss.backward()
                        self.optimizer.step()
                        # loss_per_epoch += loss.item()
                        tepoch.set_postfix(loss=f"{loss.item():.4f}")

            # Validation Loop
            self.model.eval()

            with torch.inference_mode():
                outputs = self.model(X_val)
                loss = self.loss_fn(outputs, y_val)
                print(f"Fold {i+1} MSE: {loss.item()}")

class Decision:

    def __init__(self):
        self.model = Rollout()
        # if os.path.isfile("rollout.pth"):
        #     self.model.load_state_dict(torch.load("rollout.pth", weights_only=True))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00007)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.device = torch_directml.device()
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(total_params)

    def transfer_data(self, record: list, result: list):
        state = []
        state_answer = []

        for mini_record, res in zip(record, result):
            # print(len(mini_record), len(res), res[0], res[-1])
            for idx in range(len(res)):
                # print(idx)
                mini_state = []
                for j in range(idx - 2, idx): # 4 recent moves
                    if j < 0 or j >= len(mini_record):
                        mini_state.append(np.zeros((19,19)))
                        mini_state.append(np.zeros((19,19)))
                        mini_state.append(np.zeros((19,19)))
                    else:
                        temp = np.where(mini_record[j] == -1, 1, 0)
                        mini_state.append(temp)
                        temp = np.where(mini_record[j] == 1, 1, 0)
                        mini_state.append(temp)
                        mini_state.append(mini_record[j])

                for j in range(idx - 6, idx): # 6 recent moves
                    moves = np.zeros((19,19))
                    if j < 0 or j >= len(res):
                        mini_state.append(moves)
                    else:
                        moves[res[j][1], res[j][0]] = 1
                        mini_state.append(moves)
                assert len(mini_state) == 12

                temp_board = np.zeros((19,19))
                temp_board[res[idx][1], res[idx][0]] = 1

                state_answer.append(temp_board)
                state.append(mini_state)
        
        X = torch.tensor(np.array(state, dtype=np.float32))
        y = torch.tensor(np.array(state_answer, dtype=np.float32))
        y = y.view(-1, 361)
        return X, y
    def train(self, record, result, load = True):
        def accuracy_fn(y_true, y_pred):
            correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
            acc = (correct / len(y_pred)) * 100
            return acc
        
        if load is True:
            X, y = self.transfer_data(record, result)
        else:
            X, y = torch.tensor(np.array(record, dtype=np.float32)), torch.tensor(np.array(result, dtype=np.float32))
        print(X.shape, y.shape)
        epochs = 50
        batch_size = 128
        # Split the data into training and testing sets
        X = X.to(self.device)
        y = y.to(self.device)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        coord = torch.argmax(y_train, axis=1)
        unique, counts = torch.unique(coord, return_counts=True)
        sorted_freq = sorted(zip(unique.cpu().numpy(), counts.cpu().numpy()), key=lambda x: x[1], reverse=True)[:50]
        print(f"Frequency in y_predict: {dict(sorted_freq)}")
        # print(f"Frequency in y_predict: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # for sample in y_train:
        #     print(sample)
        
        # Training Loop
        for epoch in range(epochs):
            self.model.train()
            loss_per_epoch = 0.0
            with tqdm(
                train_loader, unit="step", ncols=90, desc=f"Epoch {epoch + 1}/{epochs}") as tepoch:
                for batch_X, batch_y in tepoch:
                    self.optimizer.zero_grad()
                    y_pred = self.model(batch_X)
                    loss: torch.Tensor = self.loss_fn(y_pred, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    loss_per_epoch += loss.item()
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

        # Validation Loop
            if (epoch + 1) % 2 == 0:
                self.model.eval()
                with torch.inference_mode():
                    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=2000, shuffle=True)
                    for batch_X, batch_y in test_loader:
                        y_pred = self.model(batch_X)
                        y_predict = torch.argmax(y_pred, axis=1)
                        y_true = torch.argmax(batch_y, axis=1)
                        unique, counts = torch.unique(y_predict, return_counts=True)
                        print(f"Frequency in y_predict: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
                        # print(y_predict.shape, y_true.shape)
                        # print(y_predict[:10], y_true[:10])
                        acc = accuracy_fn(y_predict, y_true)
                        print(f"Accuracy: {acc:.4f}%")
                        break

        torch.save(self.model.state_dict(), "rollout.pth")
        return
    
if __name__ == "__main__":
    agent = Decision()
    X = np.load("X_dec.npy")
    y = np.load("y_dec.npy")
    agent.train(X, y, False)