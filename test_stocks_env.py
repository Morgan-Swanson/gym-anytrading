# Tests by Morgan Swanson
from unittest import TestCase
import pandas as pd
import numpy as np
import gym
from gym_anytrading.envs.stocks_env import StocksEnv

class TestStocksEnv(TestCase):
    def setUp(self):
        df = pd.DataFrame({'Close'  : [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
                           'Twitter': [3, 4, 5, 6, 7, 8, 5, 3, 2, 2, 2, 2, 1, 1, 1],
                           'Reddit' : [4, 4, 4, 4, 4, 7, 7, 6, 3, 2, 4, 4, 4, 4, 4]})
        window_size = 1
        start_index = window_size
        end_index = len(df)
        self.env = StocksEnv(df, window_size=window_size, frame_bound=(start_index, end_index))
        self.env2 = StocksEnv(df, window_size=window_size, frame_bound=(start_index, end_index), diffs=False)


class TestProcessData(TestStocksEnv):
    def test_step(self):
        self.assertEqual(np.array([[1, 3, 4, 0, 0, 0]]).all(), self.env.reset().all())

    def test_step(self):
        self.assertEqual(np.array([[1, 3, 4]]).all(), self.env2.reset().all())
