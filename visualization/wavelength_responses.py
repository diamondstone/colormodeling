#!/usr/bin/python
"""Simple 2d graph of SML sensitivity vs. wavelength."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import collections

def main():
  data = np.genfromtxt('data/smj10.csv', delimiter=',', names=['wavelength', 'L_response', 'M_response', 'S_response'])
  # Undo the log transform
  for key in ['L_response', 'M_response', 'S_response']:
    data[key] = map(math.exp, data[key])
  # Normalize by total response
  totals = data['L_response'] + data['M_response'] + data['S_response']
  for key in ['L_response', 'M_response', 'S_response']:
    data[key] = data[key] / totals
  # Now we have points in the triangle L + M + S = 1, L, M, S > 0. We want to
  # plot the curve in this triangle, so we compute the x and y value. Choice of
  # axes can be arbitrary, but I'll put blue in the lower left and red in the lower
  # right similar to the CIE color space.
  xs = .5 * (data['L_response'] - data['S_response'])
  ys = math.sqrt(3) / 2 * data['M_response']

  fig, ax = plt.subplots()

  fig.patch.set_visible(False)
  ax.axis('off')
  # jet colormap is a poor approximation
  # TODO: use a colormodel to compute the closest representable color to the
  # calculated cone responses given a typical RGB monitor.
  colormap = plt.cm.jet
  normalization = plt.Normalize(min(data['wavelength']), max(data['wavelength']))
  colors = colormap(normalization(data['wavelength']))

  # Ideally I would just do ax.plot(xs, ys, c=colors) here,
  # but that only accepts a single color, not an array.
  # Instead I create a set of line segments and color them by hand,
  # borrowing the technique from the scipy cookbook.
  points = np.array([xs, ys]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = collections.LineCollection(segments, colors=colors)
  lc.set_array(data['wavelength'])
  lc.set_linewidth(3)

  ax.add_collection(lc)
  # Show the triangular border
  ax.plot([-.5, 0], [0, math.sqrt(3) / 2], c='black')
  ax.plot([.5, 0], [0, math.sqrt(3) / 2], c='black')
  ax.plot([-.5, .5], [0, 0], c='black')
  # TODO: label vertices LMS
  fig.suptitle('Realized cone responses by wavelength')
  plt.show()


if __name__ == '__main__':
  main()
