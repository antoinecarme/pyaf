from CodeGen import TS_CodeGen_Objects as tscodegen

class cTimeSeriesCodeGenerator:
    
    def __init__(self):
        self.mInternalCodeGen = None;

    def generateCode(self, iAutoForecast, iDSN = None, iDialect = None):
        self.mInternalCodeGen = tscodegen.cDecompositionCodeGenObject();
        lSQL = self.mInternalCodeGen.generateCode(iAutoForecast , iDSN, iDialect);
        return lSQL;


    def testGeneration(self, iAutoForecast, iDSN = None, iDialect = None):
        lSQL = None;
        # 
        # 
        lKnownDSNs = ["sqlite://",
                      "sqlite:///a.db",
                      "postgresql://antoine:@/githubtest?port=5433",
                      "mysql://user:pass@localhost/GitHubtest",
                      ];
        for lDSN in lKnownDSNs:            
            print(" ******************************** testGeneration_start  ", lDSN , " ******************************** ");
            try:
                self.mInternalCodeGen = tscodegen.cDecompositionCodeGenObject(lDSN, iDialect);
                lSQL = self.mInternalCodeGen.testGeneration(iAutoForecast);
                print("TS_CODE_GEN_SUCCESS" , lDSN);
            except Exception as e: 
                print("TS_CODE_GEN_FAILURE" , lDSN);
                print("FAILURE_WITH_EXCEPTION : " , lDSN, str(e)[:200])
                # traceback.print_exc()
                lSQL = None;
                # raise;
            print(" ******************************** testGeneration_end  ", lDSN , " ******************************** ");
        
        
    
