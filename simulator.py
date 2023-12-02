from collections import namedtuple, Counter, defaultdict
from tqdm import tqdm
import random
import math
import re
import matplotlib.pyplot as plt
import inspect

class ChemicalReactionNetwork:
    def __init__(self, crn_string: str):
        # determine the set of species and create a fitting type to store configurations
        self.species = sorted(set(re.findall(r'[^\s\+\->]+', crn_string)))
        class Configuration(namedtuple("Configuration", self.species, defaults=(0,)*len(self.species))):
            def __add__(self, other):
                return Configuration(*(a+b for a, b in zip(self, other)))
            def __sub__(self, other):
                return Configuration(*(a-b for a, b in zip(self, other)))
        self.Configuration = Configuration

        # create a reactions dict which stores the configuration change if a reaction is applied
        self.reactions = defaultdict(lambda: Configuration())
        for reaction_string in crn_string.strip().splitlines():
            reactants, results = (tuple(s.strip() for s in half_string.split("+"))
                                  for half_string in reaction_string.split("->"))
            self.reactions[reactants] = self.Configuration(**Counter(results)) - self.Configuration(**Counter(reactants))

        # precompute 'minimal_reactants' for the is_stable method
        self._minimal_reactants = [ Configuration(**Counter(reactants)) for reactants in self.reactions ]

    def is_stable(self, config) -> bool:
        # a config is stable if there is no possible reactions
        return all(any(s < required_s for s, required_s in zip(config, required_config))
                   for required_config in self._minimal_reactants)

def run_simulation(crn: ChemicalReactionNetwork, initial_config: tuple[int, ...]) -> float:
    config = crn.Configuration(*initial_config)
    interactions = 0
    while not crn.is_stable(config):
        interactions += 1
        sample = tuple(random.sample(crn.species, counts=config, k=2))
        config += crn.reactions[sample]
    return interactions / (sum(initial_config) or 1)

def main():
    crn = ChemicalReactionNetwork("""A + B -> A + U
                                  B + A -> B + U
                                  A + U -> A + A
                                  B + U -> B + B""")
    num_simulations = 100
    ns = range(0, 101, 2)
    configs = [ lambda n: (n, 100-n, 0),
                lambda n: (n//2, n//2, 0),
                lambda n: (n//3, 2*n//3, 0),
                lambda n: (n//3, n//3, n//3),
                lambda n: (n, 20, 0),
                lambda n: (1, 0, n-1) ]

    _, axes = plt.subplots(nrows=2, ncols=math.ceil(len(configs)/2), figsize=(10*math.ceil(len(configs)/2), 20))
    for ax, config in zip(axes.flatten(), configs):
        description = ", ".join([ re.findall(r"\(.*?\)", inspect.getsource(config))[0],
                                  f"{ns.start}≤n≤{ns.stop-1}",
                                  f"step={ns.step}",
                                  f"{num_simulations} simulations" ])
        print(description)
        dataset = [ [ run_simulation(crn, config(n)) for _ in range(num_simulations) ]
                    for n in tqdm(ns) ]
        ax.boxplot(dataset)
        ax.set(title=description, xlabel="n", xticks=range(1, len(ns)+1, 5), xticklabels=ns[::5], ylabel="time")
        ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
