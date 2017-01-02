#!/usr/bin/python
"""Parametric graph of LMS responses by wavelength in the plane L+M+S=1."""

from __future__ import division

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections

### Temporary hack to allow imports from color_model
### TODO: refactor so all scripts are in root directory
### and visualization only includes
import sys
import os
sys.path.append(os.path.split(sys.path[0])[0])

from models import color_model


def _LmsToX(l, m, s):
  return .5 * (l - s)


def _LmsToY(l, m, s):
  return m * math.sqrt(3) / 2

def Interpolate(values, num):
  prev = values[0]
  yield prev
  for v in values:
    for i in xrange(1, num+1):
      yield (prev * (num - i) + v * i) / num
    prev = v


def main():
  data = np.genfromtxt(
    'data/smj10.csv', delimiter=',',
    names=['wavelength', 'L_response', 'M_response', 'S_response'])
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
  xs = _LmsToX(data['L_response'], data['M_response'], data['S_response'])
  ys = _LmsToY(data['L_response'], data['M_response'], data['S_response'])
  xs = np.array(list(Interpolate(xs, 10)))
  ys = np.array(list(Interpolate(ys, 10)))

  fig, ax = plt.subplots()

  # The lowest points on either end of the curve correspond to 380 nm (index 0)
  # and 700 nm (index -15). We will use as our base of scaling the point on the
  # line between those points, (xs[0], ys[0]) and (xs[-15], ys[-15]) with x = 0.
  # Slight fudging needed so it actually looks straight.
  base = (ys[0]*xs[-15] - ys[-15]*xs[0]) / (xs[-15]-xs[0]) - .005

  ax.axis('off')
  RGB = color_model.BEST_RGB_COLOR_MODEL
  for t in xrange(100, 0, -1):
    scale_factor = t / 100.0
    scaled_xs = xs * scale_factor
    scaled_ys = (ys - base) * scale_factor + base
    colors = []
    for i in xrange(len(xs)):
      m = scaled_ys[i] * 2 / math.sqrt(3)
      # using l + s + m = 1, and l - s = 2x we have
      l = (1 - m + 2*scaled_xs[i])/2
      s = (1 - m - 2*scaled_xs[i])/2
      actual_color = color_model.Color(l, m, s)
      rgb_coords = RGB.ApproximateRepresentation(actual_color)
      scale_factor = 1 / max(rgb_coords)
      rgb_coords = np.minimum(rgb_coords*scale_factor, np.matrix([1]*3).getT())
      colors.append(rgb_coords)

    # Ideally I would just do ax.plot(xs, ys, c=colors) here,
    # but that only accepts a single color, not an array.
    # Instead I create a set of line segments and color them by hand,
    # borrowing the technique from the scipy cookbook.
    points = np.array([scaled_xs, scaled_ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = collections.LineCollection(segments, colors=colors)
    lc.set_linewidth(3)
    ax.add_collection(lc)

  # Show the triangular border
  ax.plot([-.5, 0], [0, math.sqrt(3) / 2], c='black', linewidth=2.0)
  ax.plot([.5, 0], [0, math.sqrt(3) / 2], c='black', linewidth=2.0)
  # Fudge lower border slightly since pyplot is cutting off below 0 for some
  # dumb reason
  ax.plot([-.498, .498], [0.002, 0.002], c='black', linewidth=2.0)
  plt.text(-.53, -.04, 'S', fontsize=15)
  plt.text(.51, -.04, 'L', fontsize=15)
  plt.text(-.015, math.sqrt(3)/2+.02, 'M', fontsize=15)
  fig.suptitle('Realizable cone responses through combining colors')
  fig.set_facecolor('white')
  plt.savefig('realizable_colors.png', facecolor='white')


if __name__ == '__main__':
  main()
