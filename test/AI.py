from torch import nn, optim
import torch
import numpy as np
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
    with open(file_name, 'r') as f:
        data = eval(f.read())  # Evaluate the string as a Python object

    # Convert the nested list to a NumPy array for efficiency
    data_np = np.array(data, dtype=np.float32)
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(data_np)
    tensor.requires_grad = True
    return tensor

X_input = read_file("X_input.txt")
y_input = read_file("y_input.txt")
assert X_input.shape == (101, 3, 19, 19)
assert y_input.shape == (101, 1)

X_train = X_input[:80]
y_train = y_input[:80]
X_test = X_input[80:]
y_test = y_input[80:]
class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.conv1 = nn.Conv2d(3, 19, 3)
        # self.conv2 = nn.Conv2d(15, 13, 3)
        self.conv2 = nn.Conv2d(19, 19, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(19 * 5 * 5, 19 * 19)
        self.fc2 = nn.Linear(19 * 19, 256)
        self.out = nn.Linear(256, 1)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.tensor):
        debug = False
        x = self.activation(self.conv1(x))

        x = self.activation(self.conv2(x))

        x = self.pool(x)

        x = self.activation(self.conv2(x))

        print(x.shape) if debug else None

        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out(x)
        assert not debug
        return x


model = AI()
loss_fn = nn.MSELoss()
adam = optim.Adam(model.parameters(), lr=0.001)

torch.set_printoptions(precision=2)

def train(model, x, y, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        model.train()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_pred = model(X_test)
            test_loss = loss_fn(y_pred, y_test)
            # y_test_pred = model(X_test)
            # test_loss = loss_fn(y_test_pred, y_test)
            if (epoch + 1) % 10 == 0:
                # print(y_test_pred)
                print(y_pred)
                print(y_test)
                print(
                    f"Epoch: {epoch + 1}, Loss: {loss.item():.4f} \
                    | Test Loss: {test_loss.item():.4f}"
                )
torch.set_printoptions(profile="default")


train(model, X_input, y_input, 100, loss_fn, adam)
