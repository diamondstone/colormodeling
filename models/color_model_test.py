#!/usr/bin/python
"""Unit tests for color_model.py."""

import unittest

import numpy as np

import color_model

MAUVE = color_model.Color(.4, .3, .5)

class IdealColorModelTest(unittest.TestCase):
  def testRepresentation_Mauve(self):
    self.assertTrue(np.array_equal(np.matrix('.5; .3; .4'),
                             color_model.IDEAL_COLOR_MODEL.Representation(MAUVE)))

  def testColor_MauveRepresentation(self):
    self.assertEquals(MAUVE, color_model.IDEAL_COLOR_MODEL.Color(
                                 color_model.IDEAL_COLOR_MODEL.Representation(MAUVE)))

if __name__ == '__main__':
  unittest.main()
