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

from openquake.hazardlib.scalerel.etna import EtnaMSR
from openquake.hazardlib.tests.scalerel.msr_test import BaseMSRTestCase


class EtnaTestCase(BaseMSRTestCase):
    '''
    Tests the area-magnitude and magnitude-area scaling relationships
    derived for Mt Etna.

    '''
    MSR_CLASS = EtnaMSR

    def test_median_magnitude(self):
        """
        This tests the ASR
        """
        self._test_get_median_mag(2.667, None, 4.016, places=3)
        self._test_get_median_mag(6.000, None, 4.382, places=3)
        self._test_get_median_mag(10.667, None, 4.641, places=3)
        self._test_get_median_mag(16.667, None, 4.842, places=3)
        self._test_get_median_mag(24.000, None, 5.006, places=3)

    def test_median_area(self):
        """
        This tests the MSR
        """
        self._test_get_median_area(2.0, None, 0.0306, places=3)
        self._test_get_median_area(3.0, None, 0.282, places=3)
        self._test_get_median_area(4.0, None, 2.594, places=3)
        self._test_get_median_area(5.0, None, 23.873, places=3)
        self._test_get_median_area(6.0, None, 219.740, places=3)
