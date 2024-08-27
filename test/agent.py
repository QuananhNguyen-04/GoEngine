import sgf
import random
import os
from sklearn.model_selection import train_test_split
import torch
# import torch_directml

# import intel_extension_for_pytorch as ipex
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from AI import Eval, Rollout

torch.manual_seed(42)


class Agent:
    def __init__(self) -> None:

        self.moves = []
        # self.moves = [
        #     [16, 3], [3, 15], [15, 16], [3, 3], [2, 2], [3, 2], [2, 3], [3, 4], [1, 5], [14, 2], [15, 4], [16, 14], [13, 15], [15, 10], [11, 2], [13, 3], [8, 2], [16, 6], [14, 6], [12, 4], [15, 2], [16, 8], [2, 13], [5, 16], [2, 6], [6, 2],[7, 2],[6, 3],[8, 4],[10, 4],[4, 1],[7, 4],[8, 5],[9, 3],[6, 1],[5, 1],[5, 2],[8, 3],[7, 3],[6, 4],[9, 2],[10, 2],[10, 1],[10, 3],[8, 1],[6, 6],[8, 7],[10, 7],[6, 8],[3, 7],[3, 6],[4, 6],[10, 8],[11, 8],[10, 9],[11, 9],[5, 3],[5, 4],[9, 7],[10, 6],[10, 10],[5, 8],[6, 9],[2, 7],[1, 7],
        #     [5, 9],
        #     [6, 10], [2, 14], [4, 5], [5, 7], [5, 5], [3, 5], [6, 5], [7, 5], [4, 4], [7, 6], [4, 3],
        #     [3, 13],
        #     [11, 10],     [8, 16],     [9, 5],     [15, 5],     [3, 12],     [2, 12],
        #     [2, 11],
        #     [1, 13],     [4, 10],    [2, 10],     [2, 8],     [3, 10],    [3, 9],
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
        self.model = Eval()
        model_source = "eval.pth"
        if os.path.isfile(model_source):
            self.model.load_state_dict(
                torch.load(
                    model_source, weights_only=True, map_location=torch.device("cpu")
                )
            )
        # self.device = torch_directml.device()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0003)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(total_params)

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

    def transfer_data(
        self, records: list[np.ndarray], results: list[float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transfer data from a list of records and results to a tensor of inputs and a tensor of outputs.

        Args:
            records (List[np.ndarray]): A list of records, where each record is a 3D numpy array of shape (game_depth + 2, 19, 19)
            results (List[float]): A list of results, where each result is a float

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. The first tensor is the input tensor of shape (batch_size, 3, 19, 19),
                and the second tensor is the output tensor of shape (batch_size, 1)
        """
        inputs = []
        outputs = []
        for record, result in zip(records, results):
            game_depth = len(record)
            for idx, (state0, state1, state2) in enumerate(
                zip(record, record[1:], record[2:])
            ):
                inputs.append([state0, state1, state2])
                outputs.append(
                    torch.tanh(
                        torch.tensor(result * idx * (game_depth - idx - 1) / game_depth 
                                    + random.uniform(-2.0, 2.0) / random.randint(1, 5))
                ))

        inputs_tensor = torch.tensor(np.array(inputs, dtype=np.float32))
        outputs_tensor = torch.tensor(np.array(outputs, dtype=np.float32)).unsqueeze(1)

        return inputs_tensor, outputs_tensor

    def train(self, records, results, load: bool = True) -> None:
        """
        Train the model with the given records and results.

        Args:
            records (List[torch.Tensor] | np.ndarray): \
                A list of records, where each record is a 3D tensor of shape (game_depth + 2, 19, 19)
            results (List[torch.Tensor] | np.ndarray): \
                A list of results, where each result is a float
            load (bool, optional): Whether to load the records and results from files. Defaults to True.
        """

        model = self.model
        if load:
            inputs, targets = self.transfer_data(records, results)
            np.save("X_eval.npy", inputs.numpy())
            np.save("y_eval.npy", targets.numpy())
            print("X, y saved")
        else:
            inputs = torch.tensor(np.array(records, dtype=np.float32))
            targets = torch.tensor(np.array(results, dtype=np.float32))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        model.to(self.device)

        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            inputs, targets, test_size=0.1, random_state=42, shuffle=False
        )

        train_dataset = TensorDataset(train_inputs, train_targets)
        test_dataset = TensorDataset(test_inputs, test_targets)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        epochs = 50
        for epoch in range(epochs):
            loss_per_epoch = 0.0
            model.train()
            with tqdm(
                train_loader, unit="step", ncols=80, desc=f"Epoch {epoch + 1}/{epochs}"
            ) as tepoch:
                for batch_inputs, batch_targets in tepoch:
                    predictions = model(batch_inputs)
                    loss = self.loss_fn(predictions, batch_targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_per_epoch += loss.item()
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            if (epoch + 1) % 2 == 0:
                with torch.no_grad():
                    loss = 0.0
                    for batch_inputs, batch_targets in test_loader:
                        predictions = model(batch_inputs)
                        loss += self.loss_fn(predictions, batch_targets).item()
                    print(
                        f"Epoch: {epoch + 1} | Loss : {loss_per_epoch/len(train_loader):.4f} | Test Loss: {loss / len(test_loader):.4f}"
                    )

        model.eval()
        with torch.no_grad():
            predictions = model(test_inputs[:500])
            test_loss = self.loss_fn(predictions, test_targets[:500])
            for pred, actual in zip(predictions[20:30], test_targets[20:30]):
                print(pred.item(), actual.item())
            print(test_loss.item())
        torch.save(model.state_dict(), "eval.pth")

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
                    train_loader,
                    unit="step",
                    ncols=80,
                    desc=f"Epoch {epoch + 1}/{epochs}",
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
                for j in range(idx - 2, idx):  # 4 recent moves
                    if j < 0 or j >= len(mini_record):
                        mini_state.append(np.zeros((19, 19)))
                        mini_state.append(np.zeros((19, 19)))
                        mini_state.append(np.zeros((19, 19)))
                    else:
                        temp = np.where(mini_record[j] == -1, 1, 0)
                        mini_state.append(temp)
                        temp = np.where(mini_record[j] == 1, 1, 0)
                        mini_state.append(temp)
                        mini_state.append(mini_record[j])

                for j in range(idx - 6, idx):  # 6 recent moves
                    moves = np.zeros((19, 19))
                    if j < 0 or j >= len(res):
                        mini_state.append(moves)
                    else:
                        moves[res[j][1], res[j][0]] = 1
                        mini_state.append(moves)
                assert len(mini_state) == 12

                temp_board = np.zeros((19, 19))
                temp_board[res[idx][1], res[idx][0]] = 1

                state_answer.append(temp_board)
                state.append(mini_state)

        X = torch.tensor(np.array(state, dtype=np.float32))
        y = torch.tensor(np.array(state_answer, dtype=np.float32))
        y = y.view(-1, 361)
        return X, y

    def train(self, record, result, load=True):
        def accuracy_fn(y_true, y_pred):
            correct = (
                torch.eq(y_true, y_pred).sum().item()
            )  # torch.eq() calculates where two tensors are equal
            acc = (correct / len(y_pred)) * 100
            return acc

        if load is True:
            X, y = self.transfer_data(record, result)
            np.save("X_dec.npy", X.numpy())
            np.save("y_dec.npy", y.numpy())
        else:
            X, y = torch.tensor(np.array(record, dtype=np.float32)), torch.tensor(
                np.array(result, dtype=np.float32)
            )
        print(X.shape, y.shape)
        epochs = 50
        batch_size = 128
        # Split the data into training and testing sets
        X = X.to(self.device)
        y = y.to(self.device)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, random_state=42
        )
        coord = torch.argmax(y_train, axis=1)
        unique, counts = torch.unique(coord, return_counts=True)
        sorted_freq = sorted(
            zip(unique.cpu().numpy(), counts.cpu().numpy()),
            key=lambda x: x[1],
            reverse=True,
        )[:50]
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
                train_loader, unit="step", ncols=90, desc=f"Epoch {epoch + 1}/{epochs}"
            ) as tepoch:
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
                    test_loader = DataLoader(
                        TensorDataset(X_test, y_test), batch_size=2000, shuffle=True
                    )
                    for batch_X, batch_y in test_loader:
                        y_pred = self.model(batch_X)
                        y_predict = torch.argmax(y_pred, axis=1)
                        y_true = torch.argmax(batch_y, axis=1)
                        unique, counts = torch.unique(y_predict, return_counts=True)
                        print(
                            f"Frequency in y_predict: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}"
                        )
                        # print(y_predict.shape, y_true.shape)
                        # print(y_predict[:10], y_true[:10])
                        acc = accuracy_fn(y_predict, y_true)
                        print(f"Accuracy: {acc:.4f}%")
                        break

        torch.save(self.model.state_dict(), "rollout.pth")
        return


if __name__ == "__main__":
    # agent = Decision()
    # X = np.load("X_dec.npy")
    # y = np.load("y_dec.npy")
    # agent.train(X, y, False)
    agent = Evaluation()
    X = np.load("X_eval.npy")
    y = np.load("y_eval.npy")
    agent.train(X, y, False)