import timeit
import numpy as np
from scipy.stats import binom

def generate_binom_rvs():
    n = 100
    p = 0.5
    size = 1000
    return binom.rvs(n=n, p=p, size=size)

def generate_uniform_and_compare():
    n = 100
    p = 0.5
    size = 1000
    uniform_rvs = np.random.uniform(size=(size, n))
    return (uniform_rvs < p).sum(axis=1)

binom_time = timeit.timeit(generate_binom_rvs, number=100)
uniform_time = timeit.timeit(generate_uniform_and_compare, number=100)

print(f"Binomial RVs Time: {binom_time} seconds")
print(f"Uniform and Compare Time: {uniform_time} seconds")
