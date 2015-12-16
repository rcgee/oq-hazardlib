# The Hazard Library
# Copyright (C) 2015 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module :mod:`openquake.hazardlib.scalerel.etna` implements
:class:`EtnaMSR`.

"""
from math import log10
from openquake.hazardlib.scalerel.base import BaseASR, BaseMSR


class EtnaMSR(BaseASR, BaseMSR):
    """
    Implements magnitude-area and area-magnitude scaling relationships
    derived for Mt Etna. Original relationship determines magnitude (Ml)
    from rupture length (km):
    Ml = (3.391 +/- 0.192) + (2.076 +/- 0.414) * log10(length)

    NOTE: Rupture length is converted to area assuming an aspect ratio
    of 1.5 (lengh:width). In order to preserve the original rupture length,
    the same aspect ratio must be used in later calculations.

    See Azzaro R., D'Amico S., Pace B. and Peruzza L.; 2014: Estimating
    the expected seismicity rates of volcano-tectonic earthquakes at Mt.
    Etna (Italy) by a geometric-kinematic approach. Gruppo Nazionale di
    Geofisica della Terra Solida, Atti del 33 Convegno Nazionale, Tema 1:
    Geodinamica, 20-25.
    """
    def get_median_mag(self, area, rake):
        """
        This ASR returns magnitude (Ml) given an area (km^2), assumes
        aspect ratio = 1.5; rake and standard deviation are ignored.

        :param area:
            Area in square km.
        """
        return 3.391 + 2.076 * log10((1.5*area)**0.5)

    def get_median_area(self, mag, rake):
        """
        This MSR returns area (km^2) given a magnitude (Ml); assumes
        aspect ratio = 1.5; rake and standard deviation are ignored.

        :param magnitude:
            Magnitude (Ml)
        """
        return (2./3)*((10 ** (-1.633 + 0.482 * mag))**2)