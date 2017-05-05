import pyaf.Bench.TS_datasets as tsds

import pyaf.Bench.MComp as mcomp


#tester1 = mcomp.cMComp_Tester(tsds.load_M1_comp());
#tester1.testSignals('')
#tester1.testAllSignals()

#tester2 = mcomp.cMComp_Tester(tsds.load_M2_comp());
#tester1.testSignals('')
#tester2.testAllSignals()

#tester3 = mcomp.cMComp_Tester(tsds.load_M3_Y_comp());
#tester1.testSignals('')
#tester3.testAllSignals()

#tester4 = mcomp.cMComp_Tester(tsds.load_M3_Q_comp());
#tester1.testSignals('')
#tester4.testAllSignals()

#tester5 = mcomp.cMComp_Tester(tsds.load_M3_M_comp());
#tester1.testSignals('')
#tester5.testAllSignals()

#tester6 = mcomp.cMComp_Tester(tsds.load_M3_Other_comp());
#tester1.testSignals('')
#tester6.testAllSignals()

tester7 = mcomp.cMComp_Tester(tsds.load_M4_comp("FINANCE") , "M4COMP");
tester7.testSignals('FIN1')
# tester7.testAllSignals()
