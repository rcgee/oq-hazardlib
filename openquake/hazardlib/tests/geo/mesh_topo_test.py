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
import math

import numpy

from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from openquake.hazardlib.geo.mesh import Mesh, RectangularMesh
from openquake.hazardlib.geo import utils as geo_utils

from openquake.hazardlib.tests import assert_angles_equal
from openquake.hazardlib.tests.geo import _mesh_test_data


class _BaseMeshTestCase(unittest.TestCase):
    def _make_mesh(self, lons, lats, depths=None):
        mesh = Mesh(lons, lats, depths)
        self.assertIs(mesh.lons, lons)
        self.assertIs(mesh.lats, lats)
        self.assertIs(mesh.depths, depths)
        return mesh


class MeshCreationTestCase(_BaseMeshTestCase):
    def test_1d(self):
        mesh = self._make_mesh(numpy.array([1, 2]), numpy.array([0, 0]),
                               numpy.array([-10, -10]))
        self.assertEqual(len(mesh), 2)

    def test_from_points_list_with_depth(self):
        points = [Point(0, 1, -2), Point(2, 3, -4), Point(5, 7, -10)]
        mesh = Mesh.from_points_list(points)
        self.assertTrue((mesh.depths == [-2, -4, -10]).all())
        self.assertEqual(mesh.depths.dtype, numpy.float)


class MeshIterTestCase(_BaseMeshTestCase):
    def test_1d(self):
        mesh = self._make_mesh(numpy.array([0.1, 0.2, 0.3]),
                               numpy.array([0.9, 0.8, 0.7]),
                               numpy.array([-0.4, -0.5, -0.6]))
        self.assertEqual(list(mesh),
                         [Point(0.1, 0.9, -0.4), Point(0.2, 0.8, -0.5),
                          Point(0.3, 0.7, -0.6)])

    def test_2d(self):
        lons = numpy.array([[1.1, 2.2], [2.2, 3.3]])
        lats = numpy.array([[-7, -8], [-9, -10]])
        depths = numpy.array([[-1, -2], [-3, -4]])
        points = list(self._make_mesh(lons, lats, depths))
        self.assertEqual(points, [Point(1.1, -7, -1), Point(2.2, -8, -2),
                                  Point(2.2, -9, -3), Point(3.3, -10, -4)])


class MeshSlicingTestCase(_BaseMeshTestCase):
    def test_1d(self):
        lons = numpy.array([1, 2, 3, 4, 5, 6])
        lats = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        depths = numpy.array([-7.1, -7.2, -7.3, -7.4, -7.5, -7.6])
        mesh = self._make_mesh(lons, lats, depths)
        submesh = mesh[-4:]
        self.assertEqual(len(submesh), 4)
        self.assertTrue((submesh.lons == [3, 4, 5, 6]).all())
        self.assertTrue((submesh.lats == [0.3, 0.4, 0.5, 0.6]).all())
        self.assertTrue((submesh.depths == [-7.3, -7.4, -7.5, -7.6]).all())

        with self.assertRaises(AssertionError):
            submesh = mesh[0:0]

