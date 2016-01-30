"""Defines colors and the color model interface."""

from collections import namedtuple

import numpy as np

class Color(namedtuple('Color', ['l_response', 'm_response', 's_response'])):
  """Class for representing a color.

  I could define a color either perceptually, as a typical trichromatic
  human's cone responses, or physically, as a map from wavelengths to energies.

  There are advantages and disadvantages to each approach, but I have chosen the
  former option. This has the advantage that colors will be the same iff they
  appear the same to a typical person, but the disadvantage of
  not reflecting the physical reality, so other tests (like finding the
  spectrum) may distinguish between identical 'colors'.
  """

class ColorModel(object):
  """Interface class for color models.

  ColorModels must implement the basic functionality of mapping between colors
  and model-specific representations. Not all colors are necessarily
  representable within the model.
  """

  def IsRepresentable(self, color):
    """Determines whether a color is representable by the model.

    Should return a boolean.
    """
    raise NotImplementedError

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    The model may specify the type of a representation. Should return None
    if the color is not representable.
    """
    raise NotImplementedError

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    raise NotImplementedError


class ThreePrimaryAdditiveColorModel(ColorModel):
  """Color model of colors representable by combining three additive primaries.
  
  Representations are triples of float values indicating the energy of each
  primary to combine.
  """
  
  def __init__(self, colors):
    if len(colors) != 3:
      raise ValueError('Must be given three primary colors.')
    self._colors = colors
    self._encode_matrix = np.matrix(colors)
    try:
      self._decode_matrix = self._encode_matrix.getI()
    except np.linalg.linalg.LinAlgError:
      raise ValueError('Given primaries must have linearly independent LMS response vectors.')      

  def IsRepresentable(self, color):
    """Representable colors are positive linear combinations of primaries."""
    color_vector = np.matrix(color).getT()  # Need column vector.
    return all(x > 0 for x in self._encode_matrix * color_vector)

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    Representations are three-dimensional numpy column vectors.
    """
    if not self.IsRepresentable(color):
      return None
    color_vector = np.matrix(color).getT()  # Need column vector.
    return self._encode_matrix * color_vector

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    # decode_matrix * representation is another column vector. We need to convert
    # to an array and reshape before we can unpack it.
    return Color(*np.array(self._decode_matrix * representation).reshape(3))


# Standard RGB color model. Provides a decent model of what a standard RGB monitor can
# display, and how it displays it.
SRGB_RED = Color(0.0, 0.0, 1.0)  # TODO: find real LMS responses of sRGB red.
SRGB_GREEN = Color(0.0, 1.0, 0.0)  # TODO: find real LMS responses of sRGB green.
SRGB_BLUE = Color(1.0, 0.0, 0.0)  # TODO: find real LMS responses of sRGB blue.
SRGB_PRIMARIES = (SRGB_RED, SRGB_GREEN, SRGB_BLUE)
SRGB_COLOR_MODEL = ThreePrimaryAdditiveColorModel(SRGB_PRIMARIES)


# "Ideal" RGB color model. Can represent non-real colors (colors which cannot be
# perceived by looking at any mixture of wavelengths).
IDEAL_RED = Color(0.0, 0.0, 1.0)
IDEAL_GREEN = Color(0.0, 1.0, 0.0)
IDEAL_BLUE = Color(1.0, 0.0, 0.0)
IDEAL_PRIMARIES = (IDEAL_RED, IDEAL_GREEN, IDEAL_BLUE)
IDEAL_COLOR_MODEL = ThreePrimaryAdditiveColorModel(IDEAL_PRIMARIES)


# Useful trick:
# Every color is representable in the IDEAL_COLOR_MODEL, every IDEAL_COLOR_MODEL
# representation is also a valid SRGB_COLOR_MODEL representation.
# SRGB_COLOR_MODEL representations can easily be displayed as colors on a typical
# monitor.
# So we can convert from every color to a color we can display in a nice way using:
# SRGB_COLOR_MODEL.Color(IDEAL_COLOR_MODEL.Representation(color)).
