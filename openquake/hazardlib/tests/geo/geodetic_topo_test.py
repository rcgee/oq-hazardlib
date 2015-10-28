# The Hazard Library
# Copyright (C) 2012-2014, GEM Foundation
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
import unittest
import collections

import numpy

from openquake.hazardlib.geo import geodetic

from openquake.hazardlib.tests import speedups_on_off

Point = collections.namedtuple("Point",  'lon lat')

# these points and tests that use them are from
# http://williams.best.vwh.net/avform.htm#Example
LAX = (118 + 24 / 60., 33 + 57 / 60.)
JFK = (73 + 47 / 60., 40 + 38 / 60.)


@speedups_on_off
class TestDistance(unittest.TestCase):
    def topo_test1(self):
        p1 = (0, 0, -1)
        p2 = (0, 0, 1)
        distance = geodetic.distance(*(p1 + p2))
        self.assertAlmostEqual(distance, 2.0)

    def topo_test2(self):
        p1 = (0, 0, 1)
        p2 = (0, 0, -1)
        distance = geodetic.distance(*(p1 + p2))
        self.assertAlmostEqual(distance, 2.0)

    def topo_test3(self):
        p1 = (0, 0, -1)
        p2 = (0.5, -0.3, 1)
        distance = geodetic.distance(*(p1 + p2))
        self.assertAlmostEqual(distance, 64.8678482287)

    def topo_test4(self):
        p1 = (0, 0, 1)
        p2 = (0.5, -0.3, -1)
        distance = geodetic.distance(*(p1 + p2))
        self.assertAlmostEqual(distance, 64.8678482287)
