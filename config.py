import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA = 1
GAMMA = 50
T = .875
SEED = 17
LEARNING_RATE = 5e-4
