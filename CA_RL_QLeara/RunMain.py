import Game
import tensorflow as tf
from collections import deque
import numpy as np

def train_network():
    game=Game.Game()
    D=deque()
