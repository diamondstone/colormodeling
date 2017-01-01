#!/usr/bin/python
"""Unit tests for color_model.py."""

import unittest

import numpy as np

import color_model

MAUVE = color_model.Color(.4, .3, .5)

class IdealColorModelTest(unittest.TestCase):
  def testRepresentation_mauve_ideal(self):
    np.testing.assert_array_almost_equal(
        np.matrix('.4; .3; .5'),
        color_model.IDEAL_COLOR_MODEL.Representation(MAUVE))

  def testColorAndRepresentationAreInverses_mauve_ideal(self):
    np.testing.assert_array_almost_equal(
      MAUVE,
      color_model.IDEAL_COLOR_MODEL.Color(
          color_model.IDEAL_COLOR_MODEL.Representation(MAUVE)))

  def testRepresentation_srgbRed_srgb(self):
    np.testing.assert_array_almost_equal(
        np.matrix('1.0; 0.0; 0.0'),
        color_model.SRGB_COLOR_MODEL._AlmostRepresentation(color_model.SRGB_RED))

  def testColorAndRepresentationAreInverses_mauve_srgb(self):
    np.testing.assert_array_almost_equal(
        MAUVE,
        color_model.SRGB_COLOR_MODEL.Color(
            color_model.SRGB_COLOR_MODEL.Representation(MAUVE)))

if __name__ == '__main__':
  unittest.main()
