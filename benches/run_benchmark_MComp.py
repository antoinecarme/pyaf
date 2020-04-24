import pandas as pd
import numpy as np
# import SignalDecomposition as SigDec
from pyaf.Bench import TS_datasets as tsds
from pyaf.Bench import MComp as mcomp


#tester1 = mcomp.cMComp_Tester(tsds.load_M1_comp());
#tester1.testSignal('')
#tester1.testAllSignals()

#tester2 = mcomp.cMComp_Tester(tsds.load_M2_comp());
#tester1.testSignal('')
#tester2.testAllSignals()

#tester3 = mcomp.cMComp_Tester(tsds.load_M3_Y_comp());
#tester1.testSignal('')
#tester3.testAllSignals()

#tester4 = mcomp.cMComp_Tester(tsds.load_M3_Q_comp());
#tester1.testSignal('')
#tester4.testAllSignals()

#tester5 = mcomp.cMComp_Tester(tsds.load_M3_M_comp());
#tester1.testSignal('')
#tester5.testAllSignals()

#tester6 = mcomp.cMComp_Tester(tsds.load_M3_Other_comp());
#tester1.testSignal('')
#tester6.testAllSignals()

tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp() , "M4_COMP");
#tester7.testSignal('FIN1')
tester7.testAllSignals()
