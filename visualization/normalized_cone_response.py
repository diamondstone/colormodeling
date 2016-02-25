#!/usr/bin/python
"""Simple 2d graph of SML sensitivity vs. wavelength."""

import math

import matplotlib.pyplot as plt
import numpy as np


def main():
  data = np.genfromtxt('data/smj10.csv', delimiter=',', names=['wavelength', 'L_response', 'M_response', 'S_response'])
  for key in ['L_response', 'M_response', 'S_response']:
    data[key] = map(math.exp, data[key])
  plt.plot(data['wavelength'], data['L_response'], c='r')
  plt.plot(data['wavelength'], data['M_response'], c='g')
  plt.plot(data['wavelength'], data['S_response'], c='b')
  plt.title('Cone responses vs. wavelength, equal energies')    
  plt.xlabel('wavelength (nm)')
  plt.ylabel('Normalized cone response')
  plt.show()


if __name__ == '__main__':
  main()
