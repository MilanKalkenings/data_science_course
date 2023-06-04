import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, ttest_ind


def plot_continuous(x: np.ndarray, y: np.ndarray, title: str):
    plt.plot(x,y)
    plt.title(title)
    plt.show()


def plot_discrete(x: np.ndarray, y: np.ndarray, title: str):
    plt.stem(x, y)
    plt.title(title)
    plt.show()


# continuous
def plot_normal(mean: float = 0, std: float = 1, x: np.ndarray = np.arange(-5, 5, 0.1)):
    density = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    plot_continuous(x=x, y=density, title="normal distribution pdf")


def plot_exponential(lamb: float = 0.5, x: np.ndarray = np.arange(-5, 5, 0.1)):
    density = lamb * np.exp(-lamb * x)
    plot_continuous(x=x, y=density, title="exponential distribution pdf")


def plot_uniform(lower: float, upper: float, x: np.ndarray = np.arange(-500, 500, 1)):
    until_lower = np.argwhere(x == lower)[0][0]
    after_upper = np.argwhere(x == upper)[0][0]
    x_until_lower = x[:until_lower]
    x_after_upper = x[after_upper:]
    x_within = x[until_lower:after_upper]
    density_within = np.ones_like(x_within) / (upper - lower)
    density_lower = np.zeros_like(x_until_lower)
    density_upper = np.zeros_like(x_after_upper)
    density = np.concatenate((density_lower, density_within, density_upper))

    plot_continuous(x=x, y=density, title="uniform distribution pdf")


# discrete
def plot_binomial(n: int = 10, p: float = 0.5):
    x = np.arange(0, n+1)
    mass = binom.pmf(x, n, p)
    plot_discrete(x=x, y=mass, title="binomial distribution pmf")


# hypothesis testing
# t_test
def get_p_value(mean1: float, mean2: float, size:int, var:float=1):
    array1 = np.random.normal(mean1, var, size)
    array2 = np.random.normal(mean2, var, size)
    _, p_value = ttest_ind(array1, array2)
    return p_value


np.random.seed(1)
sizes = np.arange(start=3, step=2, stop=50)
p_values_significant = []
p_values_insignificant = []
for size in sizes:
    p_values_significant.append(get_p_value(mean1=2, mean2=1, size=size))
    p_values_insignificant.append(get_p_value(mean1=1.05, mean2=1, size=size))

plt.plot(sizes, p_values_significant, label="big mean difference")
plt.plot(sizes, p_values_insignificant, label="small mean difference")
plt.axhline(y=0.05, color='black', linestyle='--', label="threshold")
plt.ylabel("p value")
plt.xlabel("sample size")
plt.legend()
plt.show()

# same with bigger variance:
np.random.seed(1)
sizes = np.arange(start=3, step=2, stop=50)
p_values_significant = []
p_values_insignificant = []
for size in sizes:
    p_values_significant.append(get_p_value(mean1=2, mean2=1, size=size, var=2))
    p_values_insignificant.append(get_p_value(mean1=1.05, mean2=1, size=size, var=2))

plt.plot(sizes, p_values_significant, label="big mean difference")
plt.plot(sizes, p_values_insignificant, label="small mean difference")
plt.axhline(y=0.05, color='black', linestyle='--', label="threshold")
plt.ylabel("p value")
plt.xlabel("sample size")
plt.legend()
plt.show()
# bigger variance -> bigger p values
