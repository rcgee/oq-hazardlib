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


@speedups_on_off
class MinDistanceTest(unittest.TestCase):
    # test relies on geodetic.distance() to work right
    def _test(self, mlons, mlats, mdepths, slons, slats, sdepths,
              expected_mpoint_indices):
        mlons, mlats, mdepths = [numpy.array(arr, float)
                                 for arr in (mlons, mlats, mdepths)]
        slons, slats, sdepths = [numpy.array(arr, float)
                                 for arr in (slons, slats, sdepths)]
        actual_indices = geodetic.min_distance(mlons, mlats, mdepths,
                                               slons, slats, sdepths,
                                               indices=True)
        numpy.testing.assert_equal(actual_indices, expected_mpoint_indices)
        dists = geodetic.min_distance(mlons, mlats, mdepths,
                                      slons, slats, sdepths)
        expected_closest_mlons = mlons.flat[expected_mpoint_indices]
        expected_closest_mlats = mlats.flat[expected_mpoint_indices]
        expected_closest_mdepths = mdepths.flat[expected_mpoint_indices]
        expected_distances = geodetic.distance(
            expected_closest_mlons, expected_closest_mlats,
            expected_closest_mdepths,
            slons, slats, sdepths
        )
        self.assertTrue((dists == expected_distances).all())

        # testing min_geodetic_distance with the same lons and lats
        min_geod_distance = geodetic.min_geodetic_distance(mlons, mlats,
                                                           slons, slats)
        min_geo_distance2 = geodetic.min_distance(mlons, mlats, mdepths * 0,
                                                  slons, slats, sdepths * 0)
        numpy.testing.assert_almost_equal(min_geod_distance, min_geo_distance2)

    def test_one_point(self):
        mlons = numpy.array([-0.1, 0.0, 0.1])
        mlats = numpy.array([0.0, 0.0, 0.0])
        mdepths = numpy.array([0.0, -10.0, -20.0])

        self._test(mlons, mlats, mdepths, -0.05, 0.0, 0,
                   expected_mpoint_indices=0)
        self._test(mlons, mlats, mdepths, [-0.1], [0.0], [-20.0],
                   expected_mpoint_indices=1)

    def test_several_points(self):
        self._test(mlons=[10., 11.], mlats=[-40, -41], mdepths=[-10., -20.],
                   slons=[9., 9.], slats=[-39, -45], sdepths=[-0.1, -0.2],
                   expected_mpoint_indices=[0, 1])

    def test_different_shapes(self):
        self._test(mlons=[0.5, 0.7], mlats=[0.7, 0.9], mdepths=[-13., -17.],
                   slons=[-0.5] * 3, slats=[0.6] * 3, sdepths=[-0.1] * 3,
                   expected_mpoint_indices=[0, 0, 0])

    def test_rect_mesh(self):
        self._test(mlons=[[10., 11.]], mlats=[[-40, -41]], mdepths=[[-1., -2.]],
                   slons=[9., 9.], slats=[-39, -45], sdepths=[-0.1, -0.2],
                   expected_mpoint_indices=[0, 1])


class NPointsBetweenTest(unittest.TestCase):
    # values are verified using pyproj's spherical Geod
    def test(self):
        lons, lats, depths = geodetic.npoints_between(
            lon1=40.77, lat1=38.9, depth1=10,
            lon2=31.14, lat2=46.23, depth2=-2,
            npoints=7
        )
        expected_lons = [40.77, 39.316149154562076, 37.8070559966114,
                         36.23892429550906, 34.60779411051164,
                         32.90956020775102, 31.14]
        expected_lats = [38.9, 40.174608368560094, 41.43033989236144,
                         42.66557829138413, 43.87856696738466,
                         45.067397797471415, 46.23]
        expected_depths = [10, 8, 6, 4, 2, 0, -2]
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))
        # the last and the first points should be exactly the same as two
        # original corner points, so no "assertAlmostEqual" for them
        self.assertEqual(lons[0], 40.77)
        self.assertEqual(lats[0], 38.9)
        self.assertEqual(depths[0], 10)
        self.assertEqual(lons[-1], 31.14)
        self.assertEqual(lats[-1], 46.23)
        self.assertEqual(depths[-1], -2)

   # def test_same_points(self):
        lon, lat, depth = 1.2, 3.4, -5.6
        lons, lats, depths = geodetic.npoints_between(
            lon, lat, depth, lon, lat, depth, npoints=7
        )
        expected_lons = [lon] * 7
        expected_lats = [lat] * 7
        expected_depths = [depth] * 7
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))


class NPointsTowardsTest(unittest.TestCase):
    def test(self):
        lons, lats, depths = geodetic.npoints_towards(
            lon=-30.5, lat=23.6, depth=55, azimuth=-100.5,
            hdist=400, vdist=-60, npoints=5
        )
        expected_lons = [-30.5, -31.46375358, -32.42503446,
                         -33.3837849, -34.33995063]
        expected_lats = [23.6, 23.43314083, 23.26038177,
                         23.08178673, 22.8974212]
        expected_depths = [55, 40, 25, 10, -5]
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))
        # the first point should be exactly the same
        # as the original starting point
        self.assertEqual(lons[0], -30.5)
        self.assertEqual(lats[0], 23.6)
        self.assertEqual(depths[0], 55)
        self.assertAlmostEqual(lons[-1], -34.33995063)
        self.assertAlmostEqual(lats[-1], 22.8974212)
        self.assertAlmostEqual(depths[-1], -5)

    def test_zero_distance(self):
        lon, lat, depth, azimuth = 12, 34, -56, 78
        lons, lats, depths = geodetic.npoints_towards(
            lon, lat, depth, azimuth, hdist=0, vdist=0, npoints=5
        )
        expected_lons = [lon] * 5
        expected_lats = [lat] * 5
        expected_depths = [depth] * 5
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))


class IntervalsBetweenTest(unittest.TestCase):
    def test_round_down(self):
        lons, lats, depths = geodetic.intervals_between(
            lon1=10, lat1=-4, depth1=35,
            lon2=12, lat2=4, depth2=-5,
            length=380
        )
        expected_lons = [10., 10.82836972, 11.65558943]
        expected_lats = [-4, -0.68763949, 2.62486454]
        expected_depths = [35.,18.43802828, 1.87605655] 
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))

    def test_round_up(self):
        lons, lats, depths = geodetic.intervals_between(
            lon1=10, lat1=-4, depth1=35,
            lon2=12, lat2=4, depth2=-5,
            length=350
        )
        expected_lons = [10., 10.76308634, 11.52482625, 12.28955192]
        expected_lats = [-4, -0.94915589, 2.10185625, 5.15249576]
        expected_depths = [35., 19.74555236, 4.49110472, -10.76334292] 
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))

    def test_zero_intervals(self):
        lons, lats, depths = geodetic.intervals_between(
            lon1=10, lat1=1, depth1=5, lon2=10.04, lat2=1.5, depth2=-5,
            length=140
        )
        expected_lons = [10]
        expected_lats = [1]
        expected_depths = [5]
        self.assertTrue(numpy.allclose(lons, expected_lons))
        self.assertTrue(numpy.allclose(lats, expected_lats))
        self.assertTrue(numpy.allclose(depths, expected_depths))

    def test_same_number_of_intervals(self):
        # these two set of points are separated by a distance of 65 km. By
        # discretizing the line every 2 km, the number of expected points for
        # both sets (after rounding) must be 34
        lons_1, lats_1, depths_1 = geodetic.intervals_between(
            lon1=132.2272081355264675, lat1=31.0552366690758639,
            depth1=-4,
            lon2=131.6030780890111203, lat2=31.1968015468782589,
            depth2=18,
            length=2.0
        )

        lons_2, lats_2, depths_2 = geodetic.intervals_between(
            lon1=132.2218096511129488, lat1=31.0378653652772165,
            depth1=-4,
            lon2=131.5977943677305859, lat2=31.1794320218608547,
            depth2=18,
            length=2.0
        )

        self.assertTrue(34, lons_1.shape[0])
        self.assertTrue(34, lons_2.shape[0])
        self.assertTrue(34, lats_1.shape[0])
        self.assertTrue(34, lats_2.shape[0])
        self.assertTrue(34, depths_1.shape[0])
        self.assertTrue(34, depths_2.shape[0])

