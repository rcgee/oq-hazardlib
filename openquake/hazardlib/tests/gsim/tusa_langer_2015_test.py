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
#
#
#
# Test tables elaboratated from data provided directly from the authors.
#

"""
Implements the tests for the set of GMPE classes included within the
GMPE of Tusa and Langer (2015)
"""
from openquake.hazardlib.gsim.tusa_langer_2015 import (TusaLanger2015RepiBA08SE,
						       TusaLanger2015RepiBA08DE,
						       TusaLanger2015RepiSP87SE,
						       TusaLanger2015RepiSP87DE,
						       TusaLanger2015Rhypo)


from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase

# Discrepency percentages to be applied to all tests
MEAN_DISCREP = 0.1
STDDEV_DISCREP = 0.1


class TusaLanger2015RepiBA08SEtestcase(BaseGSIMTestCase):
    """
    Tests the Tusa and Langer (2015) GMPE for the case in which the BA08
    functional form is used, taking epicentral distance considering SE
    """
    
    GSIM_CLASS = TusaLanger2015RepiBA08SE
    # File containing the results for the Mean
    MEAN_FILE = "TL15/TusaLanger2015BA08SE_MEAN.csv"
    # File contaning the results for the Total Standard Deviation
    STD_FILE = "TL15/TusaLanger2015BA08SE_STD_TOTAL.csv"

    # Excel spreadsheet to calculate expected values provided by original author
    def test_mean(self):
        self.check(self.MEAN_FILE,
		   max_discrep_percentage=MEAN_DISCREP)  

    def test_std_total(self):
        self.check(self.STD_FILE,
                   max_discrep_percentage=STDDEV_DISCREP)


class TusaLanger2015RepiBA08DEtestcase(TusaLanger2015RepiBA08SEtestcase):
    """
    Tests the Tusa and Langer (2015) GMPE for the case in which the BA08
    functional form is used, taking epicentral distance considering DE
    """
    
    GSIM_CLASS = TusaLanger2015RepiBA08DE
    MEAN_FILE = "TL15/TusaLanger2015BA08DE_MEAN.csv"
    STD_FILE = "TL15/TusaLanger2015BA08DE_STD_TOTAL.csv"


class TusaLanger2015RepiSP87SEtestcase(TusaLanger2015RepiBA08SEtestcase):
    """
    Tests the Tusa and Langer (2015) GMPE for the case in which the SP87
    functional form is used, taking taking epicentral distance considering SE
    """
    
    GSIM_CLASS = TusaLanger2015RepiSP87SE
    MEAN_FILE = "TL15/TusaLanger2015SP87SE_MEAN.csv"
    STD_FILE = "TL15/TusaLanger2015SP87SE_STD_TOTAL.csv"


class TusaLanger2015RepiSP87DEtestcase(TusaLanger2015RepiBA08SEtestcase):
    """
    Tests the Tusa and Langer (2015) GMPE for the case in which the BA08
    functional form is used, taking taking epicentral distance considering DE
    """
    
    GSIM_CLASS = TusaLanger2015RepiSP87DE
    MEAN_FILE = "TL15/TusaLanger2015SP87DE_MEAN.csv"
    STD_FILE = "TL15/TusaLanger2015SP87DE_STD_TOTAL.csv"


class TusaLanger2015RhypoBA08testcase(TusaLanger2015RepiBA08SEtestcase):
    """
    Tests the Tusa and Langer (2015) GMPE for the case in which the BA08
    functional form is used, taking hypocentral distance 
    """
    
    GSIM_CLASS = TusaLanger2015Rhypo
    MEAN_FILE = "TL15/TusaLanger2015Rhypo_MEAN.csv"
    STD_FILE = "TL15/TusaLanger2015Rhypo_STD_TOTAL.csv"









