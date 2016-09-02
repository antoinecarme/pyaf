mkdir logs
ipython3 test_M1Comp.py &> logs/log.m1 &
ipython3 test_M2Comp.py &> logs/log.m2 &
ipython3 test_M3Comp.py &> logs/log.m3 &
ipython3 test_M4Comp.py &> logs/log.m4 &
ipython3 test_NN3.py &> logs/log.nn3
ipython3 test_NN5.py &> logs/log.nn5

