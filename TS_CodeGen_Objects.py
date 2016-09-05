from sqlalchemy.engine import reflection

import sqlalchemy
from sqlalchemy import *
from sqlalchemy.sql import column
from sqlalchemy.pool import NullPool

from sqlalchemy.dialects import *


from sqlalchemy.dialects.postgresql import *

from sqlalchemy.schema import DropTable
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import pandas as pd;

class date_diff(GenericFunction):
    type = Float
    
@compiles(DropTable, "postgresql")
def _compile_drop_table(element, compiler, **kwargs):
    return compiler.visit_drop_table(element) + " CASCADE"

@compiles(date_diff, 'default')
def _default_date_diff(element, compiler, **kw):  # pragma: no cover
    return "DATEDIFF(%s, %s)" % (compiler.process(element.clauses.clauses[0]),
                                 compiler.process(element.clauses.clauses[1]),
                                 )
@compiles(date_diff, 'mysql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "DATEDIFF(%s, %s)" % (compiler.process(element.clauses.clauses[0]),
                                 compiler.process(element.clauses.clauses[1]),
                                 )

@compiles(date_diff, 'postgresql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
#    return "extract(epoch from (%s - %s))::int" % (compiler.process(element.clauses.clauses[0]),
#                                              compiler.process(element.clauses.clauses[1]),
#                                     )
    return "(extract(epoch from (%s))::bigint  - extract(epoch from (%s))::bigint)" % (compiler.process(element.clauses.clauses[0]),
                                                                     compiler.process(element.clauses.clauses[1]),
                                     )

@compiles(date_diff, 'sqlite')
def _sl_date_diff(element, compiler, **kw):    # pragma: no cover
    return "julianday(%s) - julianday(%s)" % (compiler.process(element.clauses.clauses[0]),
                                              compiler.process(element.clauses.clauses[1]),
                                              )

class cDecompositionCodeGenObject:
    
    def __init__(self):
        self.mAutoForecast = None;
        self.mMeta = MetaData();
        self.mDialect = sqlalchemy.dialects.sqlite.dialect();
        pass
    
    def generate_Sql(self, statement, iDialect = None):
        if(iDialect is not None):
            return statement.compile(dialect=iDialect, compile_kwargs={'literal_binds': True}).string;
        return statement.compile(dialect=self.mDialect, compile_kwargs={'literal_binds': True}).string;
    
    def createLogicalTable(self, iTableName):
        lTestTableName = iTableName;
        lPrimaryKeyName = "PrimaryKey";
        
        lTestTable = Table(lTestTableName,
                           self.mMeta);

        lTestTable.append_column(Column(lPrimaryKeyName, Integer, primary_key=True));
        lTestTable.append_column(Column(self.mDateName, Float));
        lTestTable.append_column(Column(self.mSignalName, Float));
        lTestTable.c[ lPrimaryKeyName ].primary_key = True;
        lTestTableAlias = lTestTable.alias('ApplyDataset')
        return lTestTableAlias;

    def getDialectName(self):
        if(self.mConnection is not None):
            return self.mEngine.dialect.name;
        return "";

    def supports_CTE(self):
        #return False;
        lDialectName = self.getDialectName();
        return (("sqlite" != lDialectName) and
                ("mysql" != lDialectName));

    
    def generate_CTE(self, exprs, name):
        if(self.supports_CTE()):
            cte1 = select(exprs).cte(name)
            return cte1
        else :
            # plain sub-select
            cte1 = select(exprs).alias(name);
            return cte1

    def hasAnalyticalRowNumber(self):
        lDialectName = self.getDialectName();
        return (("sqlite" != lDialectName) and
                ("mysql" != lDialectName));

    
    def createConnection(self):
        # connected mode.
        lDSN = "sqlite://";
        # lDSN = "postgresql:///GitHubtest";
        # lDSN = "mysql://user:pass@localhost/GitHubtest";
        self.mEngine = create_engine(lDSN , echo=True)
        self.mConnection = self.mEngine.connect()
        self.mMeta = MetaData(bind = self.mConnection);
        
    def executeSQL(self, iSQL):
        stmt = alias(select([self.mModel_CTE] + []), "SQLGenResult");
        # stmt = text(iSQL);
        result = self.mConnection.execute(stmt)
        lGeneratedApplyOut = pd.DataFrame(result.fetchall());
        lGeneratedApplyOut.columns = stmt.columns.keys();
        print(lGeneratedApplyOut.head());
        return lGeneratedApplyOut;

    def testGeneration(self, iAutoForecast):
        self.createConnection();
        df = iAutoForecast.mSignalDecomposition.mBestTransformation.mSignalFrame;
        df.index.names = [ "PrimaryKey" ];
        lTestTableName = "TestTableForCodeGen";
        df.to_sql(lTestTableName , self.mConnection,  if_exists='replace', index=True)
        lSQL = self.generateCode(iAutoForecast, self.mEngine.dialect);
        self.executeSQL(lSQL);
        lTestTable = Table(lTestTableName,
                           self.mMeta,
                           autoload=True,
                           autoload_with = self.mEngine);
        lTestTable.drop();
        del self.mEngine;
        del self.mConnection;
        del self.mMeta;
        
    def getFloatLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.FLOAT);

    def getDateTimeLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.Date);


    def generateCode(self, iAutoForecast, iDialect = None):
        self.mAutoForecast = iAutoForecast;
        self.generateCode_Internal();
        statement = alias(select([self.mModel_CTE]), "SQLGenResult");
        lSQL = self.generate_Sql(statement, iDialect);
        return lSQL;

    def debrief_cte(self, cte):
        statement = select(['*']).select_from(cte);
        lSQL = self.generate_Sql(statement, None);
        print("*************************************************************************");
        print(lSQL);
        print("*************************************************************************");
        return lSQL;
        

    def generateCode_Internal(self):
         # M = T + C + AR
         # 0. the input is a table Table1 containing he date , normalized date and row number
         # 0.1 the input is a table Table1 containing he date and the signal, normalized date and row number
         # 0.2 compute the transformed sgnal in a CTE, CTE_Transformation, contains Table1
         # 1. collect trend inputs in a CTE , CTE_Trend_Inputs, containing CTE_Transformation, lag1, time, normalized time, row_number etc
         # 2. generate trend value in a CTE, CTE_Trend_value, containing Table1, time, normalized time, row number etc
         # 3. collect cycle inputs in a CTE, CTE_Cycle_Inputs, containing CTE_Trend_value and used date parts, this CTE may be the same as CTE_Trend_value
         # 4. compute trend and cycle residues in a CTE, CTE_Cycle_residues, containing CTE_Cycle_Inputs and some residues.
         # 5. compute the lags of cycle residues in a CTE, CTE_Cycle_residue_Lags, containing the previous CTE and all the lags
         # 6. compute the AR model in a CTE, CTE_AR containing CTE_Cycle_residue_Lags
         # 7. compute the model as Model = Trend + cycle + AR and the residue=  Model - Signal
         # 8. add H rows (union ?) .... WIP

         self.mTrend = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mBestModelTrend
         self.mCycle =  self.mAutoForecast.mSignalDecomposition.mBestTransformation.mBestModelCycle
         self.mAR =  self.mAutoForecast.mSignalDecomposition.mBestTransformation.mBestModelAR

         self.mSignalName = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mOriginalSignal
         self.mDateName = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTime
         self.mModelName =  self.mAutoForecast.mSignalDecomposition.mBestTransformation.mBestModelName

         self.mDateAlias = "Date" 
         self.mSignalAlias = "Signal" 
         self.mRowNumberAlias = "RN";

         self.Shortened = {};
         self.Shortened[self.mTrend.mOutName] = "STrend";
         self.Shortened[self.mCycle.mOutName] = "SCycle";
         self.Shortened[self.mAR.mOutName] = "SAR";
         self.Shortened[self.mCycle.getCycleName()] = "SCycle";
         self.Shortened[self.mCycle.getCycleResidueName()] = "SCycleRes";
         # self.Shortened[self.mModelName] = "SModel";
         print(self.Shortened);

         
         table = self.createLogicalTable("TestTableForCodeGen");
         
         self.generateRowNumberCode(table); # => RowNumber_CTE
         self.generateTransformationCode(table); # => Transformation_CTE
         self.generateTrendInputCode(); # => Trend_Inputs_CTE
         self.generateTrendCode(); # => Trend_CTE
         self.generateCycleInputCode(); # => Cycle_Inputs_CTE
         self.generateCycleCode(); # => Cycle_CTE
         self.generateCycleResidueCode(); # => Cycle_Residue_CTE
         self.generateARInputCode(); # => AR_Inputs_CTE
         self.generateARCode(); # => AR_CTE
         self.generateModelCode(); # => Model_CTE


    def addNormalizedTime(self, table):
        exprs = [];
        normalized_time = None;
        lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            lType = sqlalchemy.types.TIMESTAMP;
            # count the number of days here if date. number of seconds if datetime
            lAmplitude = func.date_diff(self.getDateTimeLiteral(lTimeInfo.mTimeMax),
                                        self.getDateTimeLiteral(lTimeInfo.mTimeMin))
            normalized_time = func.date_diff(table.c[self.mDateName],
                                             self.getDateTimeLiteral(lTimeInfo.mTimeMin));
        else:
            lAmplitude = lTimeInfo.mTimeMax - lTimeInfo.mTimeMin
            normalized_time = table.c[self.mDateName] - lTimeInfo.mTimeMin ;
        normalized_time = normalized_time / lAmplitude
        normalized_time = normalized_time.label("NTime")
        exprs = exprs + [normalized_time];
        return exprs
        
    def addRowNumber_analytical(self, table):
        exprs = [];
        row_number_column = func.row_number().over(order_by=asc(table.c[self.mDateName])) - 1
        row_number_column = row_number_column.label(self.mRowNumberAlias)
        exprs = exprs + [ row_number_column];
        return exprs

    def addRowNumber_as_count(self, table):
        exprs = [];
        TS1 = alias(select([table.columns[self.mDateName]]), "t1");
        time_expr_1 = table.c[self.mDateName];
        time_expr_2 = TS1.c[self.mDateName];
        
        expr = select([func.count(time_expr_2)]).where(time_expr_1 > time_expr_2);
        row_number_column = expr;
        row_number_column = row_number_column.label(self.mRowNumberAlias)
        exprs = exprs + [ row_number_column];
        return exprs

    def generateRowNumberCode(self, table):
        # => RowNumber_CTE
        exprs1 = None;
        if(self.hasAnalyticalRowNumber()):
            exprs1 = self.addRowNumber_analytical(table);
        else:
            exprs1 = self.addRowNumber_as_count(table);
        exprs2 = self.addNormalizedTime(table);
        self.mRowNumber_CTE = self.generate_CTE(table.columns  + exprs1 + exprs2, "RowNumber_CTE")
        self.debrief_cte(self.mRowNumber_CTE)

    def addTransformedSignal(self, table):
        exprs = []; # table.c[self.mDateName].label(self.mDateAlias), table.c[self.mSignalName].label(self.mSignalAlias)]
        trasformed_signal = table.c[ self.mSignalName ];
        trasformed_signal = trasformed_signal.label("Signal");
        exprs = exprs + [ trasformed_signal ];
        return exprs

    def generateTransformationCode(self, table):
        # => Transformation_CTE
        signal_exprs = self.addTransformedSignal(self.mRowNumber_CTE) 
        self.mTransformation_CTE = self.generate_CTE([self.mRowNumber_CTE]  + signal_exprs, "Transformation_CTE")
        self.debrief_cte(self.mTransformation_CTE)

    def julian_day(self, date_expr):
        expr_months = extract('year', date_expr) * 12 + extract('month', date_expr) 
        return expr_months;


    def addTrendInputs(self, table):
        exprs = [] ; # [table.c[self.mDateAlias], table.c[self.mSignalAlias], table.c[self.mRowNumberAlias]]
        normalized_time = table.c["NTime"];
        normalized_time_2 = normalized_time * normalized_time
        normalized_time_2 = normalized_time_2.label("NTime_2")
        normalized_time_3 = normalized_time_2 * normalized_time     
        normalized_time_3 = normalized_time_3.label("NTime_3")
        lag1 = self.getFloatLiteral(0.0);
        lag1 = lag1.label("Lag1")
        exprs = exprs + [ normalized_time, normalized_time_2, normalized_time_3, lag1]
        return exprs
    
    def generateTrendInputCode(self):
        # => Trend_Inputs_CTE
        trend_inputs = self.addTrendInputs(self.mTransformation_CTE) 
        self.mTrend_inputs_CTE = self.generate_CTE([self.mTransformation_CTE] + trend_inputs, "TICTE")
        self.debrief_cte(self.mTrend_inputs_CTE)

    def generateTrendExpression(self, table):
        trend_expr = None;
        if(self.mTrend.mFormula == "ConstantTrend"):
            trend_expr = self.getFloatLiteral(self.mTrend.mMean);
            pass
        elif(self.mTrend.mFormula == "LinearTrend"):
            print(self.mTrend.mTrendRidge.__dict__);
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"] + self.getFloatLiteral(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.mFormula == "PolyTrend"):
            print(self.mTrend.mTrendRidge.__dict__);
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"];
            trend_expr += self.mTrend.mTrendRidge.coef_[1] * table.c["NTime_2"];
            trend_expr += self.mTrend.mTrendRidge.coef_[2] * table.c["NTime_3"];
            trend_expr += self.getFloatLiteral(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.mFormula == "Lag1Trend"):
            trend_expr = table.c["Lag1"];
            pass
        return trend_expr;


    def addTrends(self, table):
        exprs = []; # [table.c[self.mDateAlias], table.c[self.mSignalAlias], table.c[self.mRowNumberAlias]]
        trend_expr = self.generateTrendExpression(table);
        #print(type(trend_expr))
        
        trend_expr = trend_expr.label(self.Shortened[self.mTrend.mOutName]);
        exprs = exprs + [trend_expr]
        return exprs

    def generateTrendCode(self):
        # => Trend_CTE
        # sel1 = select(['*']).select_from(self.mTrend_inputs_CTE)
        # alias(select([self.mTrend_inputs_CTE]), "TIN")
        # print(sel1.columns)
        trends = self.addTrends(self.mTrend_inputs_CTE)
        self.mTrend_CTE = self.generate_CTE([self.mTrend_inputs_CTE] + trends, "TCTE")
        self.debrief_cte(self.mTrend_CTE)



    def addCycleInputs(self, table):
        lTime =  self.mDateName;
        exprs = [];
        print(table.columns);
        lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            date_expr = table.c[lTime]
            date_parts = [extract('year', date_expr).label(lTime + "_Y") ,  
                          extract('month', date_expr).label(lTime + "_M") ,  
                          extract('day', date_expr).label(lTime + "_D") ,  
                          extract('hour', date_expr).label(lTime + "_h") ,  
                          extract('minute', date_expr).label(lTime + "_m") ,  
                          extract('second', date_expr).label(lTime + "_s") ,  
                          extract('dow', date_expr).label(lTime + "_dow") ,  
                          extract('week', date_expr).label(lTime + "_woy")]
            exprs = exprs + date_parts
        return exprs

    def generateCycleInputCode(self):
        # => Cycle_Inputs_CTE
        # sel1 = alias(select([self.mTrend_CTE]), "TCTE1")
        # print(sel1.columns)
        cycle_inputs = self.addCycleInputs(self.mTrend_CTE)
        self.mCycle_input_CTE = self.generate_CTE([self.mTrend_CTE] + cycle_inputs, "CICTE")
        self.debrief_cte(self.mCycle_input_CTE)


    def generateCycleExpression(self, table):
        cycle_expr = None;
        if(self.mCycle.mFormula == "NoCycle"):
            cycle_expr = self.getFloatLiteral(0.0);
            pass
        elif(self.mCycle.mFormula.startswith("Seasonal_")):
            cycle_expr = self.getFloatLiteral(0.0);
            pass
        elif(self.mCycle.mFormula.startswith("Cycle_None")):
            cycle_expr = self.getFloatLiteral(0.0);
            pass
        elif(self.mCycle.mFormula.startswith("Cycle_")):
            cycle_expr = self.getFloatLiteral(0.0);
            pass
        return cycle_expr;
    


    def addCycles(self, table):
        exprs = [];
        cycle_expr = self.generateCycleExpression(table);
        cycle_expr = cycle_expr.label(self.Shortened[self.mCycle.getCycleName()])
        exprs = exprs + [cycle_expr]
        return exprs
    
    def generateCycleCode(self):
        # => Cycle_CTE
        # sel1 = alias(select([]), "CIN")
        # print(sel1.columns)
        cycles = self.addCycles(self.mCycle_input_CTE)
        self.mCycle_CTE = self.generate_CTE([self.mCycle_input_CTE] + cycles, "CYCTE")
        self.debrief_cte(self.mCycle_CTE)


    def addCycleResidues(self, table):
        exprs = [];
        # exprs = [table]
        cycle_expr = table.c[self.Shortened[self.mCycle.getCycleName()]];
        trend_expr = table.c[self.Shortened[self.mTrend.mOutName]];
        cycle_residue_expr = trend_expr + cycle_expr - table.c[self.mSignalAlias]
        cycle_residue_expr = cycle_residue_expr.label(self.Shortened[self.mCycle.getCycleResidueName()])
        exprs = exprs + [cycle_residue_expr]
        return exprs


    def generateCycleResidueCode(self):
        # => Cycle_Residue_CTE
        # sel1 = alias(select([self.mCycle_CTE]), "CYIN")
        # print(sel1.columns)
        cycle_resdiues = self.addCycleResidues(self.mCycle_CTE)
        self.mCycle_residues_CTE = self.generate_CTE([self.mCycle_CTE] + cycle_resdiues, "CYRESCTE")
        self.debrief_cte(self.mCycle_residues_CTE)


    def createLags(self, table , H , col, index_col):
        TS = table;
        TS1 = alias(select([table.columns[index_col], table.columns[col]]), "t1");
        # TS1 = alias(table, "t");
        # TS2 = text(TS1);
        col_expr_1 = TS1.c[col];
        index_expr = TS.c[index_col]
        index_expr_1 = TS1.c[index_col]
        exprs = [];
        for h in range(1 , H+1):
            expr = select([col_expr_1]).where(index_expr == (index_expr_1 + h));
            expr = expr.label(col + "_Lag" + str(h));
            exprs = exprs + [expr];
        return exprs;

    def addARInputs(self, table):
        exprs = [];
        if(self.mAR.mFormula != "NoAR"):
            residue_name = self.Shortened[self.mCycle.getCycleResidueName()];
            exprs = exprs + self.createLags(table, 
                                            len(self.mAR.mARLagNames), 
                                            residue_name,
                                            self.mRowNumberAlias);
        return exprs


    def generateARInputCode(self):
        # => AR_Inputs_CTE
        # sel1 = alias(select([self.mCycle_residues_CTE]), "CYRESCTE1")
        # print(sel1.columns)
        ar_inputs = self.addARInputs(self.mCycle_residues_CTE)
        self.mAR_input_CTE = self.generate_CTE([self.mCycle_residues_CTE] + ar_inputs, "ARICTE")
        self.debrief_cte(self.mAR_input_CTE)
        

    def addARModel(self, table):
        exprs = [];
        ar_expr = None;
        if(self.mAR.mFormula != "NoAR"):
            i = 0 ;
            for i in range(len(self.mAR.mARLagNames)):
                feat = self.Shortened[self.mCycle.getCycleResidueName()] + "_Lag" + str(i+1);
                if(ar_expr is None):
                    ar_expr = self.mAR.mARRidge.coef_[i] * table.c[feat];
                else:
                    ar_expr = ar_expr + self.mAR.mARRidge.coef_[i] * table.c[feat];
                i = i + 1;
            ar_expr = ar_expr + self.getFloatLiteral(self.mAR.mARRidge.intercept_);
        else:
            ar_expr = self.getFloatLiteral(0.0);
        ar_expr = ar_expr.label(self.Shortened[self.mAR.mOutName])
        exprs = exprs + [ar_expr]
        return exprs

    def generateARCode(self):
        # => AR_CTE
        # sel1 = alias(select([]), "ARI")
        # print(sel1.columns)
        ars = self.addARModel(self.mAR_input_CTE)
        self.mAR_CTE = self.generate_CTE([self.mAR_input_CTE] + ars, "ARCTE")
        self.debrief_cte(self.mAR_CTE)

    def add_TS_Model(self, table):
        exprs = [];
        sum_1 = table.c[self.Shortened[self.mTrend.mOutName]];
        sum_1 += table.c[self.Shortened[self.mCycle.mOutName]];
        sum_1 += table.c[self.Shortened[self.mAR.mOutName]];
        model_expr = sum_1;
        model_expr = model_expr.label("TSModel")
        # model_residue = sum_1 - table.c[self.mSignalAlias]
        model_residue = model_expr - table.c[self.mSignalAlias]
        model_residue = model_residue.label("TSModel" + "Residue")
        exprs = exprs + [model_expr , model_residue]
        return exprs

    def generateModelCode(self):
        # => Model_CTE
        # sel1 = alias(select([]), "AR")
        # print(sel1.columns)
        model_vars = self.add_TS_Model(self.mAR_CTE)
        self.mModel_CTE = self.generate_CTE([self.mAR_CTE] + model_vars, "MODCTE")
        self.debrief_cte(self.mModel_CTE)

