import numpy as np
import pandas as pd
from bettor import Naif, Sophisticate

class Simulation:
    def __init__(self, num_naif, num_soph, p, epsilon, b_initial, mu_win):
        self.num_naif = num_naif
        self.num_soph = num_soph
        self.p = p
        self.epsilon = epsilon
        self.b_initial = b_initial
        self.mu_win = mu_win

    def run(self, T, scenario='none', delta_boost=0.5, seed=37):
        np.random.seed(seed)
        naifs = [Naif(self.p, self.epsilon, mu_win=self.mu_win) for _ in range(self.num_naif)]
        sophs = [Sophisticate(self.p, self.epsilon, mu_win=self.mu_win) for _ in range(self.num_soph)]
        records = []
        boost = 0.0

        for t in range(T):
            if scenario == 'none':
                boost = 0.0
            elif scenario == 'initial':
                boost = self.b_initial if t == 0 else 0.0
            elif scenario == 'dynamic':
                if t == 0:
                    boost = self.b_initial
                else:
                    boost = delta_boost if t % 20 == 0 else 0.0

            for bettor in naifs + sophs:
                bettor.boost = boost
                ev = bettor.perceived_ev(boost)
                if ev >= 0:
                    win = np.random.rand() < bettor.p
                    bettor.update_after_bet(win)
                records.append({
                    'time': t,
                    'type': 'Naif' if isinstance(bettor, Naif) else 'Sophisticate',
                    'betted': ev >= 0,
                    'w': bettor.w,
                    'bankroll': bettor.bankroll,
                    'boost': boost
                })

        return pd.DataFrame(records)
