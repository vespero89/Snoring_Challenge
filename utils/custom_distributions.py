import numpy as np
import scipy
import scipy.stats

class choices():
    def __init__(self, values):
        self.values = values

    def rvs(self):
        i = np.random.choice(len(self.values))
        return self.values[i]

class deterministic():
    def __init__(self, value):
        self.value = value

    def rvs(self):
        return self.value


class loguniform_gen():
    def __init__(self, base=2, low=0, high=1, round_exponent=False, round_output=False):
      self.base = base
      self.low = low
      self.high = high
      self.round_exponent = round_exponent
      self.round_output = round_output

    def rvs(self):
      exponent = scipy.stats.uniform.rvs(loc=self.low, scale=(self.high-self.low))
      if (self.round_exponent):
        exponent = np.round(exponent)

      value = np.power(self.base, exponent);
      if (self.round_output):
        value = np.round(value)

      return value;