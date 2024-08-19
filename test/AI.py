import os
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)
# X = torch.tensor(
#     [
#         [
#             [
#                 [0, 0, 0, -1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 1, 0, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                 [-1, -1, -1, -1, 1, 1, 0, 1, -1, -1, 1, -1, -1, -1, 1, 0, 0, 0, 0],
#                 [0, 1, 1, -1, -1, 1, 0, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 0, 0],
#                 [0, 0, -1, 1, 1, 1, 0, 1, -1, -1, 1, -1, 0, -1, 1, 1, -1, 1, 0],
#                 [0, 0, -1, -1, 1, 0, 1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, 1, 0],
#                 [0, 0, -1, -1, -1, 1, 0, 1, -1, 0, 0, 0, -1, 0, 0, -1, 1, 1, 0],
#                 [0, 0, 0, -1, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 1, 0],
#                 [0, -1, -1, -1, 1, 1, 0, 1, -1, -1, 0, 1, 1, 0, 0, -1, 1, 0, 1],
#                 [-1, -1, 0, -1, -1, 1, 1, -1, 0, 0, -1, -1, 1, 0, 0, -1, 1, 1, 1],
#                 [-1, 1, -1, 0, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, 0, 1, -1, 1, -1],
#                 [1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1],
#                 [1, 0, 1, -1, 0, -1, 1, 1, -1, 0, 0, 1, -1, -1, -1, 0, -1, -1, -1],
#                 [0, 1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 0, 0, -1],
#                 [1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1],
#                 [0, 0, 1, -1, 1, -1, -1, 1, 0, 0, 0, 1, -1, 1, 1, 1, 1, 1, -1],
#                 [0, 0, 1, -1, 1, -1, 1, 1, 0, -1, -1, -1, -1, -1, 1, 1, 0, 0, 1],
#                 [0, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 0],
#                 [0, 0, 0, 1, 0, 1, 1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0],
#             ]
#         ],
#         [
#             [
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
#                 [0, 1, 0, 0, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 0, -1, 1, 1],
#                 [1, 0, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 0, 1, 1, -1, -1, -1, 1],
#                 [0, 1, -1, -1, -1, -1, 1, -1, 0, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
#                 [1, -1, -1, -1, 0, -1, -1, -1, 0, 1, -1, 0, -1, 1, 0, 0, 1, 1, 1],
#                 [0, 1, 1, -1, 1, -1, 0, -1, -1, 1, 1, -1, 1, 1, 0, 1, 0, 1, -1],
#                 [1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 0, 0, 1, 1, -1],
#                 [0, 1, 1, 1, 1, 1, -1, 1, 1, 1, 0, -1, 0, -1, 1, 1, 1, -1, -1],
#                 [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 1, -1, -1, -1],
#                 [1, -1, 1, -1, 0, 1, -1, -1, 0, 0, 0, -1, 0, -1, -1, 1, -1, 0, 0],
#                 [1, -1, 1, -1, 0, 1, 1, -1, -1, -1, -1, -1, 1, -1, 0, -1, 0, -1, 0],
#                 [1, -1, 1, -1, -1, 1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 0],
#                 [-1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, -1],
#                 [0, 0, -1, -1, 1, 0, 0, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1],
#                 [0, 0, -1, 1, 1, 0, -1, 1, -1, -1, -1, -1, 1, 1, 0, 0, -1, -1, 1],
#                 [0, 0, 0, -1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, 1, -1, 1, 1],
#                 [0, 0, 0, -1, -1, 1, 0, 1, -1, -1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
#                 [0, 0, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
#                 [0, 0, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#             ]
#         ],
#         [
#             [
#                 [1, 1, 0, -1, -1, 1, 1, 1, 1, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0],
#                 [0, 0, 1, 1, -1, 1, 0, 1, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, 1],
#                 [1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 1, 0],
#                 [1, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, -1, 1, 1, 1],
#                 [1, -1, 0, 0, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 0],
#                 [-1, -1, -1, 0, -1, 0, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 0],
#                 [0, -1, 0, 0, -1, 0, -1, 1, 1, 1, 1, 0, 1, -1, -1, -1, 1, 1, 0],
#                 [1, -1, -1, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
#                 [1, -1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, -1, -1, 1, 0],
#                 [0, 1, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, -1, -1, 1],
#                 [1, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 0, 1, -1, 0, 0, -1, 1, 1],
#                 [0, 1, -1, -1, 0, -1, 1, 0, 0, 0, 0, 1, -1, -1, 1, -1, -1, 1, 0],
#                 [1, 0, 1, -1, -1, -1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 0, -1, 1, 1],
#                 [0, 1, 0, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 0, -1, -1, -1, -1],
#                 [1, 1, 1, 0, 1, 1, -1, 1, 0, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1],
#                 [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1],
#                 [0, 0, 0, -1, -1, 1, -1, 1, -1, -1, 1, 0, -1, 1, 1, 1, 1, 1, 0],
#                 [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 1],
#                 [0, 0, 0, 0, 0, 0, -1, 0, 1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0],
#             ]
#         ],
#     ],
#     dtype=torch.float32,
# )
# y = torch.tensor([[0.5], [3.5], [-1.25]])
# NewX = torch.tensor(

#     [[
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 0],
#             [1, -1, -1, 1, 0, 0, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0],
#             [-1, -1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 0, -1, 0, 0, 0],
#             [0, 0, -1, -1, 1, 1, 0, 1, -1, 0, 0, -1, -1, 1, -1, -1, 0, -1, 0],
#             [0, 0, 0, -1, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, -1, 0, -1, 0, -1],
#             [0, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1],
#             [0, 0, -1, 1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 1, 1, 1, 1, -1],
#             [0, 0, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1],
#             [0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 0, 1],
#             [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0],
#             [-1, 1, 1, 0, 1, 0, -1, 1, 1, 1, 0, -1, 1, -1, 0, 0, 0, 0, 0],
#             [-1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 1, 1, 0, 1, 0, 1],
#             [-1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
#             [-1, -1, -1, 1, 0, 0, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
#             [-1, 1, 1, 1, 0, 0, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 0],
#             [-1, -1, -1, 1, 1, 0, 1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 0],
#             [1, -1, -1, 1, 0, 0, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0],
#             [-1, -1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0],
#             [0, 0, -1, -1, 1, 1, 0, 1, -1, 0, 0, -1, -1, 1, -1, -1, 0, -1, 0],
#             [0, 0, 0, -1, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, -1, 0, -1, 0, -1],
#             [0, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1],
#             [0, 0, -1, 1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 1, 1, 1, 1, -1],
#             [0, 0, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1],
#             [0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 0, 1],
#             [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0],
#             [-1, 1, 1, 0, 1, 0, -1, 1, 1, 1, 0, -1, 1, -1, 0, 0, 0, 0, 0],
#             [-1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 1, 1, 0, 1, 0, 1],
#             [-1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
#             [-1, -1, -1, 1, 0, 0, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
#             [-1, 1, 1, 1, 0, 0, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 0],
#             [-1, -1, -1, 1, 1, 0, 1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 0],
#             [1, -1, -1, 1, 0, 0, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0],
#             [-1, -1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0],
#             [0, 0, -1, -1, 1, 1, 0, 1, -1, 0, 0, -1, -1, 1, -1, -1, 0, -1, 0],
#             [0, 0, 0, -1, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, -1, 0, -1, 0, -1],
#             [0, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1],
#             [0, 0, -1, 1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 1, 1, 1, 1, -1],
#             [0, 0, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1],
#             [0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 0, 1],
#             [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0],
#             [-1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 0, -1, 1, -1, 0, 0, 0, 0, 0],
#             [-1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 1, 1, 0, 1, 0, 1],
#             [-1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
#             [-1, -1, -1, 1, 0, 0, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
#             [-1, 1, 1, 1, 0, 0, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 0],
#             [-1, -1, -1, 1, 1, 0, 1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#     ],
#     ],
#     dtype=torch.float32,
# )
# NewY = torch.tensor([[0.75]])
# X_test = torch.tensor(
#     [
#         [
#             [
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0],
#                 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 0],
#                 [1, -1, -1, 1, 0, 0, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0],
#                 [-1, -1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0],
#                 [0, 0, -1, -1, 1, 1, 0, 1, -1, 0, 0, -1, -1, 1, -1, -1, 0, -1, 0],
#                 [0, 0, 0, -1, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, -1, 0, -1, 0, -1],
#                 [0, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1],
#                 [0, 0, -1, 1, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 1, 1, 1, 1, -1],
#                 [0, 0, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1],
#                 [0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 0, 1],
#                 [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0],
#                 [-1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 0, -1, 1, -1, 0, 0, 0, 0, 0],
#                 [-1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 1, 1, -1, 1, 1, 0, 1, 0, 1],
#                 [-1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
#                 [-1, -1, -1, 1, 0, 0, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
#                 [-1, 1, 1, 1, 0, 0, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 0],
#                 [-1, -1, -1, 1, 1, 0, 1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0],
#                 [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#             ]
#         ]
#     ],
#     dtype=torch.float32,
# )
# y_test = torch.tensor([[0.75]])


def read_file(file_name):
    # data = []
    # with open(file_name, "r", encoding="utf-8") as file:
    #     lines = file.readlines()
    #     print(lines)
    #     for line in lines:
    #         line = line.strip().strip("[]").split(",")
    #         line = [float(x) for x in line]
    #         data.append([line])
    # X = torch.tensor(data, dtype=torch.float32)
    with open(file_name, "r") as f:
        data = eval(f.read())  # Evaluate the string as a Python object

    # Convert the nested list to a NumPy array for efficiency
    data_np = np.array(data, dtype=np.float32)
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(data_np)
    tensor.requires_grad = True
    return tensor


class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.regression_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        # self.conv1 = nn.Conv2d(3, 32, 3)
        # self.conv1_1 = nn.Conv2d(32, 32, 3, padding="same")

        # self.conv2 = nn.Conv2d(32, 64, 3, 2)
        # self.conv2_1 = nn.Conv2d(64, 64, 3, padding="same")
        # self.conv2_2 = nn.Conv2d(64, 64, 3, padding="same")

        # self.conv3 = nn.Conv2d(64, 128, 3, 2)
        # self.conv3_1 = nn.Conv2d(128, 128, 3, padding="same")
        # self.conv3_2 = nn.Conv2d(128, 128, 3, padding="same")

        # self.flatten = nn.Flatten()
        
        # self.fc1 = nn.Linear(128 * 3 * 3, 512)
        # self.out = nn.Linear(512, 1)
        
        # self.activation_convo = nn.ReLU()
        # self.activation_fc = nn.Tanh()
        # self.pool = nn.MaxPool2d(2, 2)
        # self.avgpool = nn.AvgPool2d(2, 2)


    def forward(self, x: torch.tensor):
        debug = False
        # x = self.activation_convo(self.conv1(x))
        # x = self.activation_convo(self.conv3_2(x))
        # print(x.shape) if debug else None
        # x = self.activation_fc(self.fc2(x))

        x = self.block_1(x)
        print(x.shape) if debug else None
        x = self.block_2(x)
        print(x.shape) if debug else None
        x = self.block_3(x)
        print(x.shape) if debug else None
        x = self.regression_block(x)
        assert not debug
        return x


def use_model():
    model = AI()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    # if os.path.isfile("model.pth"):
    #     model.load_state_dict(torch.load("model.pth", weights_only=True))
    X_input = read_file("X_input.txt")
    y_input = read_file("y_input.txt")
    assert X_input.shape == (101, 3, 19, 19)
    assert y_input.shape == (101, 1)

    X_train = X_input[:90]
    y_train = y_input[:90]
    X_test = X_input[90:]
    y_test = y_input[90:]
    loss_fn = nn.MSELoss()
    adam = optim.Adam(model.parameters(), lr=0.0004)

    # torch.set_printoptions(precision=2)

    def train(model, x, y, epochs, loss_fn, optimizer):
        data_loader = DataLoader(
            list(zip(x, y)), batch_size=4, shuffle=True
        )
        for epoch in range(epochs):
            loss_per_epoch = 0.0
            with tqdm(data_loader, unit="step", ncols=70) as tepoch:
                for batch_X, batch_y in tepoch:
                    model.train()
                    y_pred = model(batch_X)
                    loss = loss_fn(y_pred, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_per_epoch += loss.item()
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            if (epoch + 1) % 10 == 0:
                with torch.inference_mode():
                    y_pred: torch.Tensor = model(X_test)
                    # print(y_pred.shape, y_test.shape)
                    test_loss = loss_fn(y_pred, y_test)
                    print(
                        f"Epoch: {epoch + 1} | Loss : {loss_per_epoch/len(data_loader):.4f} | Test Loss: {test_loss.item():.4f}"
                    )
        # for epoch in range(epochs):
        #     model.train()
        #     y_pred = model(x)

        #     loss = loss_fn(y_pred, y)

        #     optimizer.zero_grad()

        #     loss.backward()

        #     optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_pred = model(X_test)
            test_loss = loss_fn(y_pred, y_test)
            # y_test_pred = model(X_test)
            # test_loss = loss_fn(y_test_pred, y_test)
            print(y_pred[-5:].tolist())
            print(y_test[-5:].tolist())
            print(
                f"Epoch: {epoch + 1}, Loss: {loss.item():.4f} \
                | Test Loss: {test_loss.item():.4f}"
            )

    torch.set_printoptions(profile="default")

    train(model, X_train, y_train, 50, loss_fn, adam)

    torch.save(model.state_dict(), "model1.pth")

def eval_model():
    model = AI()
    if os.path.isfile("model1.pth"):
        model.load_state_dict(torch.load("model1.pth", weights_only=True))
    X_input = read_file("X_input.txt")
    y_input = read_file("y_input.txt")
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_input)
        print(y_pred[:10].tolist())
        print(y_input[:10].tolist())
        loss_fn = nn.L1Loss()
        MAE = loss_fn(y_pred, y_input).item()
        loss_fn = nn.MSELoss()
        MSE = loss_fn(y_pred, y_input).item()
        print(MAE, MSE)
if __name__ == "__main__":
    # use_model()
    eval_model()

class Rollout(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_0 = nn.Sequential(
            nn.Conv2d(12, 32, 7, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.decision = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 361),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        debug = False
        x = self.block_0(x)
        print(x.shape) if debug else None
        x = self.block_1(x)
        print(x.shape) if debug else None
        x = self.block_2(x)
        print(x.shape) if debug else None
        x = self.block_3(x)
        x = self.decision(x)
        print(x.shape) if debug else None
        assert not debug
        return x
