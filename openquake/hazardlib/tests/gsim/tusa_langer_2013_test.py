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

from openquake.hazardlib.gsim.tusa_langer_2013 import TusaLanger2013
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase
import numpy

if TusaLanger2013.REQUIRES_DISTANCES == {'rhypo'}:
    raise Exception, 'change GMPE back to repi and the test tables will pass'
       

class TusaLanger2013TestCase(BaseGSIMTestCase):
    GSIM_CLASS = TusaLanger2013

    # Tables provided by original authors

    def test_mean(self):
        self.check('TL13/TusaLanger2013_MEAN.csv',
                    max_discrep_percentage=0.4)

    def test_std_total(self):
        self.check('TL13/TusaLanger2013_STD_TOTAL.csv',
                    max_discrep_percentage=0.1)


