import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        pw = self.profit_win(boost)
        pl = self.profit_loss()
        return perc_p * pw + (1-perc_p) * pl

class Sophisticate(Bettor):
    def profit_win(self, boost):
        return 1 - self.p - self.epsilon + boost

    def profit_loss(self):
        return -(self.p + self.epsilon)

    def perceived_ev(self, boost):
        perc_p = self.p
        pw = self.profit_win(boost)
        pl = self.profit_loss()
        return perc_p * pw + (1-perc_p) * pl

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
        prev_frac_naif = prev_frac_soph = 1.0

        for t in range(T):
            if scenario == 'none':
                boost = 0.0
            elif scenario == 'initial':
                boost = self.b_initial if t == 0 else 0.0
            elif scenario == 'dynamic':
                if t == 0:
                    boost = self.b_initial
                else:
                    boost = delta_boost if t % 20 == 0 else 0.0 # TODO: Modify this to something else. Maybe experiment with total player count

            bets_naif = 0
            for bettor in naifs:
                bettor.boost = boost
                ev = bettor.perceived_ev(boost)
                if ev >= 0:
                    bets_naif += 1
                    win = np.random.rand() < bettor.p
                    bettor.update_after_bet(win)
                records.append({
                    'time': t,
                    'type': 'Naif',
                    'betted': ev >= 0,
                    'w': bettor.w,
                    'bankroll': bettor.bankroll,
                    'boost': boost
                })

            bets_soph = 0
            for bettor in sophs:
                bettor.boost = boost
                ev = bettor.perceived_ev(boost)
                if ev >= 0:
                    bets_soph += 1
                    win = np.random.rand() < bettor.p
                    bettor.update_after_bet(win)
                records.append({
                    'time': t,
                    'type': 'Sophisticate',
                    'betted': ev >= 0,
                    'w': bettor.w,
                    'bankroll': bettor.bankroll,
                    'boost': boost
                })

            prev_frac_naif = bets_naif / self.num_naif
            prev_frac_soph = bets_soph / self.num_soph

        return pd.DataFrame(records)

# --- Plotting function ---
# FROM CHATGPT

def plot_results(df, title):
    summary = df.groupby(['time', 'type']).agg(
        frac_bet=('betted', 'mean'),
        avg_bankroll=('bankroll', 'mean')
    ).reset_index()

    # Fraction Betting plot
    plt.figure()
    for bettor in summary['type'].unique():
        sub = summary[summary['type'] == bettor]
        plt.plot(sub['time'], sub['frac_bet'], label=bettor)
    plt.xlabel('Time')
    plt.ylabel('Fraction Betting')
    plt.title(f'{title}: Fraction Betting')
    plt.legend()
    plt.show()

    # Average Bankroll plot
    plt.figure()
    for bettor in summary['type'].unique():
        sub = summary[summary['type'] == bettor]
        plt.plot(sub['time'], sub['avg_bankroll'], label=bettor)
    plt.xlabel('Time')
    plt.ylabel('Average Bankroll')
    plt.title(f'{title}: Average Bankroll')
    plt.legend()
    plt.show()

# --- Run and plot scenarios ---

sim = Simulation(num_naif=100, num_soph=100, p=0.5, epsilon=0.1, b_initial=0.5, mu_win=0.1)
df_none = sim.run(T=500, scenario='none')
df_initial = sim.run(T=500, scenario='initial')
df_dynamic = sim.run(T=500, scenario='dynamic')

plot_results(df_none, 'No Boost')
plot_results(df_initial, 'Initial Boost')
plot_results(df_dynamic, 'Dynamic Boost')
