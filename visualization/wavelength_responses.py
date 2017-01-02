#!/usr/bin/python
"""Parametric graph of LMS responses by wavelength in the plane L+M+S=1."""

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
  interpolated_xs = np.array(list(Interpolate(xs, 10)))
  interpolated_ys = np.array(list(Interpolate(ys, 10)))

  fig, ax = plt.subplots()

  ax.axis('off')

  # Set up labels
  for i in xrange(len(data)):
    wavelength = data['wavelength'][i]
    if wavelength in {380.0, 720.0, 760.0}: continue
    if wavelength % 20 == 0 or (420 <= wavelength < 520 and wavelength % 10 == 0):
      x = xs[i]
      y = ys[i]
      slope = (x - xs[i+1]) / (ys[i+1] - y)  # Orthogonal to local slope
      rotation = math.atan(slope) * 180 / math.pi
      plt.plot(x, y)
      if wavelength < 410:
        x_offset = -.05 * 1 / (1 + slope**2)**0.5 - .008
        y_offset = -.05 * slope / (1 + slope**2)**0.5 + .02
      if 410 <= wavelength < 530:
        x_offset = -.03 * 1 / (1 + slope**2)**0.5 - .02
        y_offset = -.03 * slope / (1 + slope**2)**0.5 + .02
      if 530 <= wavelength < 690:
        x_offset = .02 * 1 / (1 + slope**2)**0.5 - .008
        y_offset = .02 * slope / (1 + slope**2)**0.5 + .02
      if 690 <= wavelength:
        x_offset = -.05 * 1 / (1 + slope**2)**0.5 - .008
        y_offset = -.05 * slope / (1 + slope**2)**0.5 + .02
      plt.annotate(int(wavelength), xy=(x, y), xytext=(x+x_offset, y+y_offset), fontsize=7,
                   rotation=rotation)

  # Set up colors
  RGB = color_model.BEST_RGB_COLOR_MODEL
  colors = []
  for i in xrange(len(interpolated_xs)):
    m = interpolated_ys[i] * 2 / math.sqrt(3)
    # using l + s + m = 1, and l - s = 2x we have
    l = (1 - m + 2*interpolated_xs[i])/2
    s = (1 - m - 2*interpolated_xs[i])/2
    actual_color = color_model.Color(l, m, s)
    rgb_coords = RGB.ApproximateRepresentation(actual_color)
    scale_factor = 1 / max(rgb_coords)
    rgb_coords = np.minimum(rgb_coords*scale_factor, np.matrix([1]*3).getT())
    colors.append(rgb_coords)

  # Ideally I would just do ax.plot(xs, ys, c=colors) here,
  # but that only accepts a single color, not an array.
  # Instead I create a set of line segments and color them by hand,
  # borrowing the technique from the scipy cookbook.
  points = np.array([interpolated_xs, interpolated_ys]).T.reshape(-1, 1, 2)
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
  fig.suptitle('Realized cone responses by wavelength')
  fig.set_facecolor('white')
  plt.savefig('realized_cone_responses.png', facecolor='white')


if __name__ == '__main__':
  main()
