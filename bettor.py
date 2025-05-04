import numpy as np
from abc import ABC, abstractmethod

class Bettor(ABC):
    def __init__(self, p, epsilon, boost=0.0, mu_win=0.0):
        self.p = p
        self.epsilon = epsilon
        self.boost = boost
        self.mu_win = mu_win
        self.w = np.random.normal(0, .5)
        self.bankroll = 0.0

    @abstractmethod
    def profit_win(self, boost):
        pass

    @abstractmethod
    def profit_loss(self):
        pass

    @abstractmethod
    def perceived_ev(self, boost):
        pass

    def update_after_bet(self, win):
        if win:
            self.bankroll += self.profit_win(self.boost)
            self.w = abs(np.random.normal(self.mu_win, .5))
        else:
            self.bankroll += self.profit_loss()
            self.w = abs(np.random.normal(0, .5))


class Naif(Bettor):
    def profit_win(self, boost):
        return 1 - self.p - self.epsilon + boost

    def profit_loss(self):
        return -(self.p + self.epsilon)

    def perceived_ev(self, boost):
        perc_p = np.clip(self.p + self.w, 0, 1)
        return perc_p * self.profit_win(boost) + (1 - perc_p) * self.profit_loss()


class Sophisticate(Bettor):
    def profit_win(self, boost):
        return 1 - self.p - self.epsilon + boost

    def profit_loss(self):
        return -(self.p + self.epsilon)

    def perceived_ev(self, boost):
        perc_p = self.p
        return perc_p * self.profit_win(boost) + (1 - perc_p) * self.profit_loss()
