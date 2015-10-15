# The Hazard Library
# Copyright (C) 2015, GEM Foundation
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
from openquake.hazardlib.gsim.montalva_2015 import (
    MontalvaEtAl2015SInter,
    MontalvaEtAl2015SSlab)
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase


class MontalvaEtAl2015SInterTestCase(BaseGSIMTestCase):
    """
    Tests the Montalva et al. (2015) GMPE for subduction
    interface earthquakes
    """
    GSIM_CLASS = MontalvaEtAl2015SInter
    MEAN_FILE = "mont15/MONTALVA_SINTER_MEAN.csv"
    TOTAL_FILE = "mont15/MONTALVA_SINTER_TOTAL.csv"
    INTER_FILE = "mont15/MONTALVA_SINTER_INTER_EVENT.csv"
    INTRA_FILE = "mont15/MONTALVA_SINTER_INTRA_EVENT.csv"

    def test_mean(self):
        self.check(self.MEAN_FILE,
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check(self.TOTAL_FILE,
                   max_discrep_percentage=0.1)

    def test_std_inter(self):
        self.check(self.INTER_FILE,
                   max_discrep_percentage=0.1)

    def test_std_intra(self):
        self.check(self.INTRA_FILE,
                   max_discrep_percentage=0.1)


class MontalvaEtAl2015SSlabTestCase(MontalvaEtAl2015SInterTestCase):
    """
    Tests the Montalva et al. (2015) GMPE for subduction inslab earthquakes
    """
    GSIM_CLASS = MontalvaEtAl2015SSlab
    MEAN_FILE = "mont15/MONTALVA_SSLAB_MEAN.csv"
    TOTAL_FILE = "mont15/MONTALVA_SSLAB_TOTAL.csv"
    INTER_FILE = "mont15/MONTALVA_SSLAB_INTER_EVENT.csv"
    INTRA_FILE = "mont15/MONTALVA_SSLAB_INTRA_EVENT.csv"
