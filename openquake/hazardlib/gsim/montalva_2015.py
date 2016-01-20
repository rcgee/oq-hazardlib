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
Module exports :class:`MontalvaEtAl2015SInter`
               :class:`MontalvaEtAl2015SSlab`
"""

from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.imt import PGA
from openquake.hazardlib.gsim.abrahamson_2015 import (AbrahamsonEtAl2015SInter,
                                                      AbrahamsonEtAl2015SSlab)


class MontalvaEtAl2015SInter(AbrahamsonEtAl2015SInter):
    """
    Adaptation of the Abrahamson et al. (2015) BC Hydro subduction interface
    GMPE, calibrated to Chilean strong motion data
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type and for PGA
        C = self.COEFFS[imt]
        C_PGA = self.COEFFS[PGA()]
        dc1_pga = C_PGA["DC1"]
        # compute median pga on rock (vs30=1000), needed for site response
        # term calculation
        pga1000 = np.exp(
            self._compute_pga_rock(C_PGA, dc1_pga, sites, rup, dists))
        mean = (self._compute_magnitude_term(C, C["DC1"], rup.mag) +
                self._compute_distance_term(C, rup.mag, dists) +
                self._compute_focal_depth_term(C, rup) +
                self._compute_forearc_backarc_term(C, sites, dists) +
                self._compute_site_response_term(C, sites, pga1000))
        stddevs = self._get_stddevs(C, stddev_types, len(sites.vs30))
        return mean, stddevs

    def _compute_magnitude_term(self, C, dc1, mag):
        """
        Computes the magnitude scaling term given by equation (2)
        """
        base = C['theta1'] + (C['theta4'] * dc1)
        dmag = self.CONSTS["C1"] + dc1
        if mag > dmag:
            f_mag = (C['theta5'] * (mag - dmag)) +\
                C['theta13'] * ((10. - mag) ** 2.)

        else:
            f_mag = (C['theta4'] * (mag - dmag)) +\
                C['theta13'] * ((10. - mag) ** 2.)

        return base + f_mag

    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1)
        """
        return (C['theta2'] + C['theta3'] * (mag - 7.8)) *\
            np.log(dists.rrup + self.CONSTS['c4'] * np.exp((mag - 6.) *
                   self.CONSTS['theta9'])) + (C['theta6'] * dists.rrup)

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt             DC1    vlin        b        theta1         theta2       theta3        theta4        theta5        theta6   theta7   theta8       theta10       theta11       theta12       theta13       theta14  theta15  theta16          phi          tau        sigma      phi_s2s
    pga     0.200000000   865.1   -1.186   4.421043841   -1.235838621  0.260594468  -1.187387838  -0.626477898  -0.000269572   1.0988    -1.42   2.949670501  -0.003104918   0.957459320  -0.157989551  -0.292014223   0.9969    -1.00  0.491926896  0.487829138  0.692798195  0.349471043
    0.020   0.200000000   865.1   -1.186   3.874508669   -1.129927615  0.303210981  -1.016431740  -0.856042843  -0.001156550   1.0988    -1.42   2.651058217  -0.001679006   0.981962173  -0.109534873  -0.223302533   0.9969    -1.00  0.504685072  0.499148597  0.709828390  0.354392334
    0.050   0.200000000  1053.5   -1.346   8.521361235   -2.153485903  0.216539087  -0.892069941  -0.441095957   0.004014115   1.2536    -1.65   3.914113094  -0.002306697   1.218408927  -0.152272689  -0.423692594   1.1030    -1.18  0.516824000  0.530025557  0.740293278  0.376100230
    0.075   0.200000000  1085.7   -1.471   9.507016024   -2.303567593  0.168972703  -0.756637072  -0.160138054   0.004278902   1.4175    -1.80   3.368267955  -0.002944423   1.466426198  -0.162075167  -0.305179119   1.2732    -1.36  0.534946941  0.575119919  0.785449648  0.415034755
    0.100   0.200000000  1032.5   -1.624   9.174229254   -2.131225355  0.092702626  -0.771497698   0.297092771   0.002695241   1.3997    -1.80   3.477428636  -0.002798884   1.599181772  -0.210800665  -0.335654541   1.3042    -1.36  0.575331833  0.544835290  0.792371259  0.466376358
    0.150   0.200000000   877.6   -1.931   7.207831450   -1.664408639  0.223043970  -1.061288150  -0.310593429   0.000966240   1.3582    -1.69   1.624196362  -0.003116267   1.855741636  -0.165407775  -0.014220063   1.2600    -1.30  0.552832400  0.546997931  0.777708427  0.426323266
    0.200   0.200000000   748.2   -2.188   4.561677306   -1.041788833  0.226168552  -0.807295794  -0.363817426  -0.002510223   1.1648    -1.49   2.627179088  -0.003119800   1.998898516  -0.132808155  -0.240953984   1.2230    -1.25  0.565973072  0.466786338  0.733631382  0.405296582
    0.250   0.200000000   654.3   -2.381   4.810358393   -1.110859712  0.227009222  -0.768610256  -0.419759393  -0.001236571   0.9940    -1.30   3.239867866  -0.000243437   2.207236399  -0.131514239  -0.398923534   1.1600    -1.17  0.539262168  0.503796884  0.737980343  0.375292128
    0.300   0.200000000   587.1   -2.518   3.006548966   -0.673431383  0.281246059  -0.954544501  -0.728037681  -0.003404116   0.8821    -1.18   3.728941071   0.002175644   2.355466769  -0.122198212  -0.548990755   1.0500    -1.06  0.575397737  0.454036959  0.732961197  0.415387646
    0.400   0.143682921   503.0   -2.657   1.453570870   -0.297213443  0.306654104  -1.064440434  -1.006169987  -0.004863754   0.7046    -0.98   3.773638124   0.000146167   2.364457383  -0.130406377  -0.582630195   0.8000    -0.78  0.558531802  0.436491292  0.708859945  0.372736555
    0.500   0.100000000   456.6   -2.669   3.307586194   -0.815724589  0.341411999  -1.407863235  -0.896819300  -0.000615682   0.5799    -0.82   3.335502796  -0.000200400   2.377150769  -0.155356690  -0.512422157   0.6620    -0.62  0.533055290  0.476332422  0.714870980  0.331627116
    0.600   0.073696559   430.3   -2.599   5.327549932   -1.343581321  0.311768348  -1.180697661  -0.749471035   0.002513322   0.5021    -0.70   4.343207653   0.000605325   2.318952991  -0.149976235  -0.730883194   0.5800    -0.50  0.527077142  0.465105269  0.702946104  0.331169660
    0.750   0.041503750   410.5   -2.401   5.589219343   -1.441383553  0.301580764  -1.382196596  -0.704555670   0.003818032   0.3687    -0.54   5.158520076   0.006268798   2.083206269  -0.192499135  -0.938076275   0.4800    -0.34  0.544745579  0.462604736  0.714668376  0.341107592
    1.000   0.000000000   400.0   -1.955   6.833915630   -1.839918841  0.139211855  -0.615908920   0.018849747   0.006807460   0.1746    -0.34   7.583151773   0.008447436   1.627620072  -0.204966366  -1.443206849   0.3300    -0.14  0.560830181  0.480690122  0.738643003  0.366107585
    1.500  -0.058496250   400.0   -1.025   6.698635097   -1.958789333  0.148294995  -0.951008651  -0.092476579   0.008414586  -0.0820    -0.05   7.640205186   0.006957710   0.829497462  -0.254484399  -1.457578682   0.3100     0.00  0.534843737  0.430881175  0.686816140  0.349243635
    2.000  -0.100000000   400.0   -0.299   5.116920971   -1.696151120  0.208255744  -0.965679548  -0.372756399   0.007452801  -0.2821     0.12   8.474441461   0.006094473   0.045703748  -0.233394032  -1.609662697   0.3000     0.00  0.523010126  0.373031102  0.642410923  0.306715602
    2.500  -0.155033971   400.0    0.000   2.952223207   -1.247457793  0.330951170  -1.364354124  -1.034194183   0.004668068  -0.4108     0.25   6.539745117   0.002886707  -0.277064931  -0.216626666  -1.193108681   0.3000     0.00  0.531096964  0.352462534  0.637411816  0.322722003
    3.000  -0.200000000   400.0    0.000   2.151641944   -1.081513741  0.348370635  -1.499087128  -1.250128425   0.003296688  -0.4466     0.30   5.935473694   0.002722442  -0.256283303  -0.229623442  -1.077893684   0.3000     0.00  0.527722253  0.354614799  0.635800622  0.347317253
    4.000  -0.200000000   400.0    0.000   1.868673247   -1.053014667  0.328124701  -1.752940627  -1.430457238   0.002985483  -0.4344     0.30   6.055671528   0.002430183  -0.362763376  -0.286285202  -1.092252871   0.3000     0.00  0.529919168  0.357819355  0.639413025  0.335891892
    5.000  -0.200000000   400.0    0.000   1.534511025   -1.096577020  0.338646912  -1.426486307  -1.363581787   0.002951891  -0.4368     0.30   5.461045784   0.003143436  -0.391044896  -0.238484997  -0.966497823   0.3000     0.00  0.542113572  0.368657975  0.655588154  0.356525842
    6.000  -0.200000000   400.0    0.000   1.937770195   -1.224379809  0.262085586  -1.023980198  -1.138124238   0.002748822  -0.4586     0.30   5.249951895   0.004509412  -0.431170579  -0.238032714  -0.939031684   0.3000     0.00  0.548589140  0.373138146  0.663462222  0.356247776
    7.500  -0.200000000   400.0    0.000   2.535966020   -1.385887157  0.194894324  -0.905332513  -0.987484435   0.003310315  -0.4433     0.30   5.894671637   0.004853253  -0.312098228  -0.271717995  -1.087338293   0.3000     0.00  0.592245922  0.395033858  0.711903773  0.431311836
    10.00  -0.200000000   400.0    0.000   2.882913552   -1.578520114  0.161250323  -1.042710824  -0.492859538   0.004355431  -0.4828     0.30   5.793722700   0.004376783  -0.261759431  -0.302832003  -1.079092370   0.3000     0.00  0.593304693  0.379251210  0.704160450  0.438551963
    """)

    CONSTS = {
        # Period-Independent Coefficients (Table 2)
        'n': 1.18,
        'c': 1.88,
        'c4': 10.0,
        'C1': 7.8,
        'theta9': 0.4
        }


class MontalvaEtAl2015SSlab(AbrahamsonEtAl2015SSlab):
    """
    Adaptation of the Abrahamson et al. (2015) BC Hydro subduction in-slab
    GMPE, calibrated to Chilean strong motion data
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type and for PGA
        C = self.COEFFS[imt]
        # For inslab GMPEs the correction term is fixed at -0.3
        dc1 = -0.3
        C_PGA = self.COEFFS[PGA()]
        # compute median pga on rock (vs30=1000), needed for site response
        # term calculation
        pga1000 = np.exp(
            self._compute_pga_rock(C_PGA, dc1, sites, rup, dists))
        mean = (self._compute_magnitude_term(C, dc1, rup.mag) +
                self._compute_distance_term(C, rup.mag, dists) +
                self._compute_focal_depth_term(C, rup) +
                self._compute_forearc_backarc_term(C, sites, dists) +
                self._compute_site_response_term(C, sites, pga1000))
        stddevs = self._get_stddevs(C, stddev_types, len(sites.vs30))
        return mean, stddevs

    def _compute_magnitude_term(self, C, dc1, mag):
        """
        Computes the magnitude scaling term given by equation (2)
        corrected by a local adjustment factor
        """
        base = C['theta1'] + (C['theta4'] * dc1)
        dmag = self.CONSTS["C1"] + dc1
        if mag > dmag:
            f_mag = (C['theta5'] * (mag - dmag)) +\
                C['theta13'] * ((10. - mag) ** 2.)

        else:
            f_mag = (C['theta4'] * (mag - dmag)) +\
                C['theta13'] * ((10. - mag) ** 2.)

        return base + f_mag

    def _compute_distance_term(self, C, mag, dists):
        """
        Computes the distance scaling term, as contained within equation (1b)
        """
        return ((C['theta2'] + C['theta14'] + C['theta3'] *
                (mag - 7.8)) * np.log(dists.rhypo + self.CONSTS['c4'] *
                np.exp((mag - 6.) * self.CONSTS['theta9'])) +
                (C['theta6'] * dists.rhypo)) + C["theta10"]

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt             DC1    vlin        b        theta1         theta2       theta3        theta4        theta5        theta6   theta7   theta8       theta10       theta11       theta12       theta13       theta14  theta15  theta16          phi          tau        sigma      phi_s2s
    pga    -0.300000000   865.1   -1.186   4.421043841   -1.235838621  0.260594468  -1.187387838  -0.626477898  -0.000269572   1.0988    -1.42   2.949670501  -0.003104918   0.957459320  -0.157989551  -0.292014223   0.9969    -1.00  0.491926896  0.487829138  0.692798195  0.349471043
    0.020  -0.300000000   865.1   -1.186   3.874508669   -1.129927615  0.303210981  -1.016431740  -0.856042843  -0.001156550   1.0988    -1.42   2.651058217  -0.001679006   0.981962173  -0.109534873  -0.223302533   0.9969    -1.00  0.504685072  0.499148597  0.709828390  0.354392334
    0.050  -0.300000000  1053.5   -1.346   8.521361235   -2.153485903  0.216539087  -0.892069941  -0.441095957   0.004014115   1.2536    -1.65   3.914113094  -0.002306697   1.218408927  -0.152272689  -0.423692594   1.1030    -1.18  0.516824000  0.530025557  0.740293278  0.376100230
    0.075  -0.300000000  1085.7   -1.471   9.507016024   -2.303567593  0.168972703  -0.756637072  -0.160138054   0.004278902   1.4175    -1.80   3.368267955  -0.002944423   1.466426198  -0.162075167  -0.305179119   1.2732    -1.36  0.534946941  0.575119919  0.785449648  0.415034755
    0.100  -0.300000000  1032.5   -1.624   9.174229254   -2.131225355  0.092702626  -0.771497698   0.297092771   0.002695241   1.3997    -1.80   3.477428636  -0.002798884   1.599181772  -0.210800665  -0.335654541   1.3042    -1.36  0.575331833  0.544835290  0.792371259  0.466376358
    0.150  -0.300000000   877.6   -1.931   7.207831450   -1.664408639  0.223043970  -1.061288150  -0.310593429   0.000966240   1.3582    -1.69   1.624196362  -0.003116267   1.855741636  -0.165407775  -0.014220063   1.2600    -1.30  0.552832400  0.546997931  0.777708427  0.426323266
    0.200  -0.300000000   748.2   -2.188   4.561677306   -1.041788833  0.226168552  -0.807295794  -0.363817426  -0.002510223   1.1648    -1.49   2.627179088  -0.003119800   1.998898516  -0.132808155  -0.240953984   1.2230    -1.25  0.565973072  0.466786338  0.733631382  0.405296582
    0.250  -0.300000000   654.3   -2.381   4.810358393   -1.110859712  0.227009222  -0.768610256  -0.419759393  -0.001236571   0.9940    -1.30   3.239867866  -0.000243437   2.207236399  -0.131514239  -0.398923534   1.1600    -1.17  0.539262168  0.503796884  0.737980343  0.375292128
    0.300  -0.300000000   587.1   -2.518   3.006548966   -0.673431383  0.281246059  -0.954544501  -0.728037681  -0.003404116   0.8821    -1.18   3.728941071   0.002175644   2.355466769  -0.122198212  -0.548990755   1.0500    -1.06  0.575397737  0.454036959  0.732961197  0.415387646
    0.400  -0.300000000   503.0   -2.657   1.453570870   -0.297213443  0.306654104  -1.064440434  -1.006169987  -0.004863754   0.7046    -0.98   3.773638124   0.000146167   2.364457383  -0.130406377  -0.582630195   0.8000    -0.78  0.558531802  0.436491292  0.708859945  0.372736555
    0.500  -0.300000000   456.6   -2.669   3.307586194   -0.815724589  0.341411999  -1.407863235  -0.896819300  -0.000615682   0.5799    -0.82   3.335502796  -0.000200400   2.377150769  -0.155356690  -0.512422157   0.6620    -0.62  0.533055290  0.476332422  0.714870980  0.331627116
    0.600  -0.300000000   430.3   -2.599   5.327549932   -1.343581321  0.311768348  -1.180697661  -0.749471035   0.002513322   0.5021    -0.70   4.343207653   0.000605325   2.318952991  -0.149976235  -0.730883194   0.5800    -0.50  0.527077142  0.465105269  0.702946104  0.331169660
    0.750  -0.300000000   410.5   -2.401   5.589219343   -1.441383553  0.301580764  -1.382196596  -0.704555670   0.003818032   0.3687    -0.54   5.158520076   0.006268798   2.083206269  -0.192499135  -0.938076275   0.4800    -0.34  0.544745579  0.462604736  0.714668376  0.341107592
    1.000  -0.300000000   400.0   -1.955   6.833915630   -1.839918841  0.139211855  -0.615908920   0.018849747   0.006807460   0.1746    -0.34   7.583151773   0.008447436   1.627620072  -0.204966366  -1.443206849   0.3300    -0.14  0.560830181  0.480690122  0.738643003  0.366107585
    1.500  -0.300000000   400.0   -1.025   6.698635097   -1.958789333  0.148294995  -0.951008651  -0.092476579   0.008414586  -0.0820    -0.05   7.640205186   0.006957710   0.829497462  -0.254484399  -1.457578682   0.3100     0.00  0.534843737  0.430881175  0.686816140  0.349243635
    2.000  -0.300000000   400.0   -0.299   5.116920971   -1.696151120  0.208255744  -0.965679548  -0.372756399   0.007452801  -0.2821     0.12   8.474441461   0.006094473   0.045703748  -0.233394032  -1.609662697   0.3000     0.00  0.523010126  0.373031102  0.642410923  0.306715602
    2.500  -0.300000000   400.0    0.000   2.952223207   -1.247457793  0.330951170  -1.364354124  -1.034194183   0.004668068  -0.4108     0.25   6.539745117   0.002886707  -0.277064931  -0.216626666  -1.193108681   0.3000     0.00  0.531096964  0.352462534  0.637411816  0.322722003
    3.000  -0.300000000   400.0    0.000   2.151641944   -1.081513741  0.348370635  -1.499087128  -1.250128425   0.003296688  -0.4466     0.30   5.935473694   0.002722442  -0.256283303  -0.229623442  -1.077893684   0.3000     0.00  0.527722253  0.354614799  0.635800622  0.347317253
    4.000  -0.300000000   400.0    0.000   1.868673247   -1.053014667  0.328124701  -1.752940627  -1.430457238   0.002985483  -0.4344     0.30   6.055671528   0.002430183  -0.362763376  -0.286285202  -1.092252871   0.3000     0.00  0.529919168  0.357819355  0.639413025  0.335891892
    5.000  -0.300000000   400.0    0.000   1.534511025   -1.096577020  0.338646912  -1.426486307  -1.363581787   0.002951891  -0.4368     0.30   5.461045784   0.003143436  -0.391044896  -0.238484997  -0.966497823   0.3000     0.00  0.542113572  0.368657975  0.655588154  0.356525842
    6.000  -0.300000000   400.0    0.000   1.937770195   -1.224379809  0.262085586  -1.023980198  -1.138124238   0.002748822  -0.4586     0.30   5.249951895   0.004509412  -0.431170579  -0.238032714  -0.939031684   0.3000     0.00  0.548589140  0.373138146  0.663462222  0.356247776
    7.500  -0.300000000   400.0    0.000   2.535966020   -1.385887157  0.194894324  -0.905332513  -0.987484435   0.003310315  -0.4433     0.30   5.894671637   0.004853253  -0.312098228  -0.271717995  -1.087338293   0.3000     0.00  0.592245922  0.395033858  0.711903773  0.431311836
    10.00  -0.300000000   400.0    0.000   2.882913552   -1.578520114  0.161250323  -1.042710824  -0.492859538   0.004355431  -0.4828     0.30   5.793722700   0.004376783  -0.261759431  -0.302832003  -1.079092370   0.3000     0.00  0.593304693  0.379251210  0.704160450  0.438551963
    """)
