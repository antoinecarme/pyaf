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
        # "sqlite://",
        # "sqlite:///a.db",
        lKnownDSNs = ["postgresql://antoine:@/githubtest?port=5433",
                      "mysql://user:pass@localhost/GitHubtest",
                      ];
        for lDSN in lKnownDSNs:            
            try:
                self.mInternalCodeGen = tscodegen.cDecompositionCodeGenObject(lDSN, iDialect);
                lSQL = self.mInternalCodeGen.testGeneration(iAutoForecast);
                print("TS_CODE_GEN_SUCCESS" , lDSN);
            except:
                print("TS_CODE_GEN_FAILURE" , lDSN);
                lSQL = None;
                raise;
        
        
    
