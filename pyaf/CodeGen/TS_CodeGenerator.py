# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

from . import TS_CodeGen_Objects as tscodegen
import traceback

class cTimeSeriesCodeGenerator:
    
    def __init__(self):
        self.mInternalCodeGen = None;

    def generateCode(self, iAutoForecast, iDSN = None, iDialect = None):
        # self.mInternalCodeGen = tscodegen.cDecompositionCodeGenObject();
        lSQL = ""; # self.mInternalCodeGen.generateCode(iAutoForecast , iDSN, iDialect);
        return lSQL;


    def testGeneration(self, iAutoForecast, iDSN = None, iDialect = None):
        lSQL = None;
        # 
        # 
        lKnownDSNs = ["postgresql://db:db@localhost/db?port=5432",
                      "oracle+cx_oracle://db:db@xe",
                      "mysql://user:pass@localhost/GitHubtest",
                      # "sqlite://",
                      # "sqlite:///a.db",
                      ];
        for lDSN in lKnownDSNs:            
            print(" ******************************** testGeneration_start  ", lDSN , " ******************************** ");
            try:
                self.mInternalCodeGen = tscodegen.cDecompositionCodeGenObject(lDSN, iDialect);
                lSQL = self.mInternalCodeGen.testGeneration(iAutoForecast);
                print("TS_CODE_GEN_SUCCESS" , lDSN);
            except Exception as e: 
                print("TS_CODE_GEN_FAILURE" , lDSN , str(e)[:200]);
                print("FAILURE_WITH_EXCEPTION : " , lDSN, str(e)[:200])
                traceback.print_exc()
                lSQL = None;
                raise;
            print(" ******************************** testGeneration_end  ", lDSN , " ******************************** ");
        
        
    
