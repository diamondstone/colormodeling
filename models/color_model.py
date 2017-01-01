"""Defines colors and the color model interface."""

from __future__ import division

from collections import namedtuple

import numpy as np

class Color(namedtuple('Color', ['l_response', 'm_response', 's_response'])):
  """Class for representing a color.

  I could define a color either perceptually, as a typical trichromatic
  human's cone responses (the "tristimulus" values), or physically,
  as a map from wavelengths to energies.

  There are advantages and disadvantages to each approach, but I have chosen the
  former option. This has the advantage that colors will be the same iff they
  appear the same to a typical person, but the disadvantage of
  not reflecting the physical reality, so other tests (like finding the
  spectrum) may distinguish between identical 'colors'.
  """


# Fudge-factor for numerical calculations. Better to return an approximate match for
# a color which can't be exactly matched, rather than not return any match.
_EPSILON = .001


WAVELENGTH_RESPONSES = np.genfromtxt(
  'data/smj10.csv', delimiter=',',
  names=['wavelength', 'L_response', 'M_response', 'S_response'])


def ColorFromWavelength(wavelength):
  """Map a single wavelength to the tristimulus responses, assuming power = 1."""
  if wavelength < 380 or wavelength > 770:
    raise ValueError("%d is not a wavelength of visible light." % wavelength)
  index = (wavelength - 380) // 5
  lower_color = Color(WAVELENGTH_RESPONSES[index][1],
                      WAVELENGTH_RESPONSES[index][2],
                      WAVELENGTH_RESPONSES[index][3])
  if wavelength == 770:
    return lower_color
  upper_color = Color(WAVELENGTH_RESPONSES[index + 1][1],
                      WAVELENGTH_RESPONSES[index + 1][2],
                      WAVELENGTH_RESPONSES[index + 1][3])
  upper_power = (wavelength % 5) / 5
  lower_power = 1-upper_power
  return _Combine(lower_color, lower_power, upper_color, upper_power)


def _Combine(color_A, power_A, color_B, power_B):
  """Mix two colors with appropriate powers."""
  return Color(color_A.l_response*power_A + color_B.l_response*power_B,
               color_A.m_response*power_A + color_B.m_response*power_B,
               color_A.s_response*power_A + color_B.s_response*power_B)


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

  def Approximation(self, color):
    """Find and the return closest representable color to the given one."""
    raise NotImplementedError


class TristimulusColorModel(ColorModel):
  """Color model of typical (non-colorblind) human color perception.
  
  Representations are tristimulus values, using the same Color class,
  so transformations are all identities.
  """
  
  def IsRepresentable(self, color):
    """All colors are representable in this model."""
    return True

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    Identity function.
    """
    return color

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    return representation

  def Approximation(self, color):
    """Find and the return closest representable color to the given one."""
    return color

  
class ThreeAdditivePrimaryColorModel(ColorModel):
  """Color model of colors representable by combining three additive primaries.
  
  Representations are triples of float values indicating the energy of each
  primary to combine.
  """
  
  def __init__(self, colors):
    if len(colors) != 3:
      raise ValueError('Must be given three primary colors.')
    self._colors = colors
    self._decode_matrix = np.matrix(colors).getT()
    try:
      self._encode_matrix = self._decode_matrix.getI()
    except np.linalg.linalg.LinAlgError:
      raise ValueError('Given primaries must have linearly independent LMS response vectors.')      

  def _AlmostRepresentation(self, color):
    """How much of the primaries would one need to combine to get the color.
    Components may be negative for non-representable colors."""
    # Need column vector.
    return self._encode_matrix * np.matrix(color).getT()
    
  def IsRepresentable(self, color):
    """Representable colors are positive linear combinations of primaries."""
    return all(x > -_EPSILON for x in self._AlmostRepresentation(color))

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    Representations are three-dimensional numpy column vectors.
    """
    if not self.IsRepresentable(color):
      return None
    return self.ApproximateRepresentation(color)

  def ApproximateRepresentation(self, color):
    """Map the given color to its representation within the model.
    
    Representations are three-dimensional numpy column vectors.
    """
    return np.maximum(self._AlmostRepresentation(color),
                      np.matrix([0]*3).getT())

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    # decode_matrix * representation is another column vector. We need to convert
    # to an array and reshape before we can unpack it.
    return Color(*np.array(self._decode_matrix * representation).reshape(3))

  def Approximation(self, color):
    """Find and the return closest representable color to the given one."""
    return self.Color(self.ApproximateRepresentation(color))


def _Normalize(lms_responses):
  """Given lms responses, return the normalized responses where sum = 1."""
  normalization_factor = 1.0 / sum(lms_responses)
  return [normalization_factor * response for response in lms_responses]


class TwoArbitraryWavelengthsColorModel(ColorModel):
  """Color model which can represent any perceptible color by combining
  two wavelengths of light at the right power.
  
  Most colors can't be realized with a single wavelength of light.
  Either they're not saturated enough, or they're some shade of purple.
  However every perceptible color can be realized by adding appropriate powers of two
  """

  def __init__(self):
    # In the plane of normalized responses, the wavelength responses form an inverted
    # u shape. If you take any point inside this u, and draw a line through it
    # parallel to the line between the tips of the u, it will cross the u in exactly two places.
    # This gives an algorithm for finding two wavelengths which can be combined to match
    # some realizable LMS response.
    # To simplify this computation, we store
    # 1) the slope between the colors at the endpoints, and
    # 2) the wavelength of the point furthest from that line.
    normalized_violet_response = _Normalize(WAVELENGTH_RESPONSES[0][1:4])
    normalized_red_response = _Normalize(WAVELENGTH_RESPONSES[-1][1:4])
    self._critical_slope = (normalized_red_response[0]-normalized_violet_response[0]) / (
        normalized_red_response[1]-normalized_violet_response[1])  
    self._greenest_wavelength = 500  # FIXME

  def IsRepresentable(self, color):
    """Representable colors are perceptible colors, so this model can determine whether a
    given color is perceptible."""
    return self.Representation(color) is not None

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    Representations are dictionaries {wavelength: power} with at most two keys.
    """
    raise NotImplementedError

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    raise NotImplementedError

  def Approximation(self, color):
    """Find and the return closest representable color to the given one."""
    raise NotImplementedError


class CIE1931ColorModel(ColorModel):
  """CIE 1931 color model. Represent a color with the xyY in CIE color space.

  xyY values are related to XYZ values by:
  x = X/(X+Y+Z)
  y = Y/(X+Y+Z)
  or, in reverse,
  X = Y*x/y
  Z = Y*(1-x-y)/y.
  """
  
  def __init__(self):
    raise NotImplementedError

  def IsRepresentable(self, color):
    """Representable colors have nonzero m response."""
    raise NotImplementedError

  def Representation(self, color):
    """Map the given color to its representation within the model.
    
    Representations are three-tuples (x, y, Y).
    """
    raise NotImplementedError

  def Color(self, representation):
    """Map the given representation to the corresponding color."""
    # decode_matrix * representation is another column vector. We need to convert
    # to an array and reshape before we can unpack it.
    raise NotImplementedError

  def Approximation(self, color):
    """Find and the return closest representable color to the given one."""
    raise NotImplementedError


# Standard RGB color model. Provides a decent model of what a standard RGB monitor can
# display, and how it displays it.
SRGB_RED = Color(0.6, 0.3, 0.1)  # TODO: find real LMS responses of sRGB red.
SRGB_GREEN = Color(0.4, 0.5, 0.1)  # TODO: find real LMS responses of sRGB green.
SRGB_BLUE = Color(0.2, 0.2, 0.6)  # TODO: find real LMS responses of sRGB blue.
SRGB_PRIMARIES = (SRGB_RED, SRGB_GREEN, SRGB_BLUE)
SRGB_COLOR_MODEL = ThreeAdditivePrimaryColorModel(SRGB_PRIMARIES)


# Best possible RGB color model with actual colors of light.
BEST_RGB_RED = ColorFromWavelength(700)
BEST_RGB_GREEN = ColorFromWavelength(530)
BEST_RGB_BLUE = ColorFromWavelength(400)
BEST_RGB_PRIMARIES = (BEST_RGB_RED, BEST_RGB_GREEN, BEST_RGB_BLUE)
BEST_RGB_COLOR_MODEL = ThreeAdditivePrimaryColorModel(SRGB_PRIMARIES)


# "Ideal" RGB color model. Can represent non-real colors (colors which cannot be
# perceived by looking at any mixture of wavelengths).
IDEAL_RED = Color(1.0, 0.0, 0.0)
IDEAL_GREEN = Color(0.0, 1.0, 0.0)
IDEAL_BLUE = Color(0.0, 0.0, 1.0)
IDEAL_PRIMARIES = (IDEAL_RED, IDEAL_GREEN, IDEAL_BLUE)
IDEAL_COLOR_MODEL = ThreeAdditivePrimaryColorModel(IDEAL_PRIMARIES)

