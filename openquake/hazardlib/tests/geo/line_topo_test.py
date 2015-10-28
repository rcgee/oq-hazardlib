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

from openquake.hazardlib import geo


class LineResampleTestCase(unittest.TestCase):
    def test_topo_resample(self):
        p1 = geo.Point(0.0, 0.0, 0.0)
        p2 = geo.Point(0.0, 0.127183341091, -14.1421356237)
        p3 = geo.Point(0.134899286793, 0.262081472606, -35.3553390593)

        resampled = geo.Line([p1, p2, p3]).resample(10.0)

        p1 = geo.Point(0.0, 0.0, 0.0)
        p2 = geo.Point(0.0, 0.0635916705456, -7.07106781187)
        p3 = geo.Point(0.0, 0.127183341091, -14.1421356237)
        p4 = geo.Point(0.0449662998195, 0.172149398777, -21.2132034356)
        p5 = geo.Point(0.0899327195183, 0.217115442616, -28.2842712475)
        p6 = geo.Point(0.134899286793, 0.262081472606, -35.3553390593)

        expected = geo.Line([p1, p2, p3, p4, p5, p6])
        self.assertEqual(expected, resampled)


class LineResampleToNumPointsTestCase(unittest.TestCase):
    def test_topo_simple(self):
        points = [geo.Point(0, 0, 1), geo.Point(0.1, 0.3, -1)]

        line = geo.Line(points).resample_to_num_points(3)
        expected_points = [geo.Point(0, 0, 1), geo.Point(0.05, 0.15, 0),
                           geo.Point(0.1, 0.3, -1)]
        self.assertEqual(line.points, expected_points)

        line = geo.Line(points).resample_to_num_points(4)
        expected_points = [geo.Point(0, 0, 1), geo.Point(0.0333333, 0.1, 0.3333),
                           geo.Point(0.0666666, 0.2, -0.3333), geo.Point(0.1, 0.3, -1)]
        self.assertEqual(line.points, expected_points)


class LineLengthTestCase(unittest.TestCase):
    def test(self):
        line = geo.Line([geo.Point(0, 0, -1), geo.Point(0, 0, 0), geo.Point(0, 0, 1)])
        length = line.get_length()
        expected_length = line.points[0].distance(line.points[1]) \
                          + line.points[1].distance(line.points[2])
        self.assertEqual(length, expected_length)
