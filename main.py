from sim import Simulation
from plot import plot_results

if __name__ == "__main__":
    sim = Simulation(num_naif=100, num_soph=100, p=0.5, epsilon=0.1, b_initial=0.5, mu_win=0.1)

    df_none = sim.run(T=500, scenario='none')
    df_initial = sim.run(T=500, scenario='initial')
    df_dynamic = sim.run(T=500, scenario='dynamic')

    plot_results(df_none, 'No Boost')
    plot_results(df_initial, 'Initial Boost')
    plot_results(df_dynamic, 'Dynamic Boost')
