import sys
import string
import random

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

import psycopg2 as psy;
#register_adapter, AsIs
from psycopg2.extensions import adapt, register_adapter, AsIs
    
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
    return "extract(epoch from (%s - %s))::int" % (compiler.process(element.clauses.clauses[0]),
                                                   compiler.process(element.clauses.clauses[1]));

@compiles(date_diff, 'sqlite')
def _sl_date_diff(element, compiler, **kw):    # pragma: no cover
    return "julianday(%s) - julianday(%s)" % (compiler.process(element.clauses.clauses[0]),
                                              compiler.process(element.clauses.clauses[1]),
                                              )
class weekday(GenericFunction):
    type = Float

@compiles(weekday, 'mysql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "WEEKDAY(%s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(weekday, 'postgresql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(dow from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(weekday, 'sqlite')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(dow, %s)" % (compiler.process(element.clauses.clauses[0]))

class week(GenericFunction):
    type = Float

@compiles(week, 'mysql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "WEEK(%s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(week, 'postgresql')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(week from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(week, 'sqlite')
def _my_date_diff(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(week, %s)" % (compiler.process(element.clauses.clauses[0]))



def cast_point(value, cur):
    if value is None:
        return None
    return value;

class cDatabaseBackend:
    
    def __init__(self, iDSN = None, iDialect = None):
        print("CREATING_DATABASE_BACKEND_DSN_DIALECT", sqlalchemy.__version__, iDSN, iDialect);
        self.mDSN = iDSN;
        self.mDialectString = iDialect;
        
        # firebird  __init__.py  mssql  mysql  oracle  postgresql  sqlite  sybase
        self.KNOWN_DIALECTS = {};
        self.KNOWN_DIALECTS [ "firebird" ] = sqlalchemy.dialects.firebird.dialect()
        self.KNOWN_DIALECTS [ "mssql" ] = sqlalchemy.dialects.mssql.dialect()
        self.KNOWN_DIALECTS [ "mysql" ] = sqlalchemy.dialects.mysql.dialect()
        self.KNOWN_DIALECTS [ "oracle" ] = sqlalchemy.dialects.oracle.dialect()
        self.KNOWN_DIALECTS [ "postgresql" ] = sqlalchemy.dialects.postgresql.dialect()
        self.KNOWN_DIALECTS [ "sqlite" ] = sqlalchemy.dialects.sqlite.dialect()
        self.KNOWN_DIALECTS [ "sybase" ] = sqlalchemy.dialects.sybase.dialect()

        self.mMeta = None;
        self.mEngine = None;
        self.mConnection = None;
        self.mDialect = None;

        self.initializeEngine();

        pass

    def initializeEngine(self):
        if(self.mDSN is not None):
            # connected mode.
            self.mEngine = create_engine(self.mDSN , echo=False)
            self.mConnection = self.mEngine.connect()
            self.mMeta = MetaData(bind = self.mConnection);
            self.mDialect = self.mEngine.dialect;
        else:
            self.mMeta = MetaData();
            self.mEngine = None;
            self.mConnection = None;
            if(self.mDialectString is not None):
                self.mDialect = self.KNOWN_DIALECTS[self.mDialectString];
    
    def generate_Sql(self, statement):
        return statement.compile(dialect=self.mDialect, compile_kwargs={'literal_binds': True}).string;
    
    def createLogicalTable(self, iTableName, iDateName, iSignalName, iDateType):
        lTestTableName = iTableName;
        lPrimaryKeyName = iDateName;
        
        lTestTable = Table(lTestTableName,
                           self.mMeta);

        # lTestTable.append_column(Column(lPrimaryKeyName, Integer, primary_key=True));
        lTestTable.append_column(Column(iDateName, iDateType));
        lTestTable.append_column(Column(iSignalName, Float));
        lTestTable.c[ lPrimaryKeyName ].primary_key = True;
        lTestTableAlias = lTestTable.alias('ApplyDataset')
        return lTestTableAlias;

    def getDialectName(self):
        if(self.mConnection is not None):
            return self.mEngine.dialect.name;
        return self.mDialectString;

    def isConnected(self):
        return (self.mConnection is not None);

    def isNewMySQL(self):
        lDialectName = self.getDialectName();
        if(self.mConnection is None):
            # risky , but no connection anyway !!!
            return True;
        if("mysql" != lDialectName):
            return False;
        (lMajor, lMinor) = self.mConnection.connection.connection._server_version;
        if((lMajor > 10) or (lMajor >= 10) and (lMinor >= 2)):
            print("MYSQL_DATABASE_SUPPORTS_CTE_AND_ROW_NUMBER" , str(lMajor) +"." + str(lMinor));
            return True;
        return False;

    def supports_CTE(self):
        #return False;
        lDialectName = self.getDialectName();
        if(("sqlite" == lDialectName) or (("mysql" == lDialectName) and not self.isNewMySQL())):
            return False;
        return True;

    
    def generate_CTE(self, exprs, name):
        if(self.supports_CTE()):
            cte1 = select(exprs).cte(name)
            return cte1
        else :
            # plain sub-select
            subselect1 = alias(select(exprs), name);
            return subselect1

    def hasAnalyticalRowNumber(self):
        lDialectName = self.getDialectName();
        if(("sqlite" == lDialectName) or (("mysql" == lDialectName) and not self.isNewMySQL())):
            return False;
        return True;

    
    def executeSQL(self, iSQL):
        stmt = text(iSQL);
        result = self.mConnection.execute(stmt)
        lGeneratedApplyOut = pd.DataFrame(result.fetchall());
        return lGeneratedApplyOut;

    def generateRandomTableName(self, length = 8):
        #        return "test_table";
        chars = string.ascii_uppercase + string.digits;
        lPrefix = "TS_CODEGEN_";
        lRandomChars = ''.join(random.choice(chars) for _ in range(length))
        return lPrefix + lRandomChars;


    def testGeneration(self, iDataFrame, iSQL, iTableName):
        lTestTableName = iTableName;
        iDataFrame.to_sql(lTestTableName , self.mConnection,  if_exists='replace', index=False)
        lTestTable = Table(lTestTableName,
                           self.mMeta,
                           autoload=True,
                           autoload_with = self.mEngine);
        lGeneratedApplyOut = self.executeSQL(iSQL);
        lTestTable.drop();
        return lGeneratedApplyOut;
        
    def getFloatLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.FLOAT);

    def getDateTimeLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.DateTime);

    def debrief_cte(self, cte):
        statement = select(['*']).select_from(cte);
        lSQL = self.generate_Sql(statement);
        print("*************************************************************************");
        print(lSQL);
        print("*************************************************************************");
        return lSQL;

    def generateSQLForStatement(self, cte):
        statement = alias(select([cte]), "SQLGenResult");
        lSQL = self.generate_Sql(statement);
        return lSQL;


'''
************************************************************************************************************************



***********************************************************************************************************************88

'''

class cDecompositionCodeGenObject:
    def __init__(self, iDSN = None, iDialect = None):
        self.mAutoForecast = None;
        self.mBackEnd = cDatabaseBackend(iDSN , iDialect);
    
    def generateCode(self, iAutoForecast, iTableName = None):
        lTableName = iTableName;
        if(lTableName is None):
            lTableName = self.mBackEnd.generateRandomTableName();
        self.mAutoForecast = iAutoForecast;
        self.generateCode_Internal(lTableName);
        lSQL = self.mBackEnd.generateSQLForStatement(self.mModel_CTE);
        return lSQL;

    def testGeneration(self, iAutoForecast, iTableName = None):
        df = iAutoForecast.mSignalDecomposition.mTrainingDataset;
        lTableName = iTableName;
        if(lTableName is None):
            lTableName = self.mBackEnd.generateRandomTableName();
        lSQL = self.generateCode(iAutoForecast, lTableName);
        lGeneratedApplyOut = self.mBackEnd.testGeneration(df , lSQL, lTableName);
        select1 = select([self.mModel_CTE]);
        lGeneratedApplyOut.columns = select1.columns.keys();
        print(lGeneratedApplyOut.head());

    def generateCode_Internal(self, iTableName = None):
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

         self.mDateAlias = "DateAlias" 
         self.mSignalAlias = "SignalAlias" 
         self.mRowNumberAlias = "RN";

         self.mDateType = sqlalchemy.types.FLOAT
         lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
         if(lTimeInfo.isPhysicalTime()):
             self.mDateType = sqlalchemy.types.TIMESTAMP;

         self.Shortened = {};
         self.Shortened[self.mTrend.mOutName] = "STrend";
         self.Shortened[self.mCycle.mOutName] = "SCycle";
         self.Shortened[self.mAR.mOutName] = "SAR";
         self.Shortened[self.mCycle.getCycleName()] = "SCycle";
         self.Shortened[self.mCycle.getCycleResidueName()] = "SCycleRes";
         # self.Shortened[self.mModelName] = "SModel";
         print(self.Shortened);

         lTableName = iTableName;
         if(lTableName is None):
             lTableName = self.mBackEnd.generateRandomTableName();
         table = self.mBackEnd.createLogicalTable(lTableName , self.mDateName, self.mSignalName, self.mDateType);
         
         self.generateRowNumberCode(table); # => RowNumber_CTE
         self.generateTransformationInputCode(table); # => Transformation_CTE
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
            lAmplitude = func.date_diff(lTimeInfo.mTimeMax,
                                        lTimeInfo.mTimeMin)
            normalized_time = func.date_diff(table.c[self.mDateName],
                                             lTimeInfo.mTimeMin);
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
        if(self.mBackEnd.hasAnalyticalRowNumber()):
            exprs1 = self.addRowNumber_analytical(table);
        else:
            exprs1 = self.addRowNumber_as_count(table);
        exprs2 = self.addNormalizedTime(table);
        self.mRowNumber_CTE = self.mBackEnd.generate_CTE(table.columns  + exprs1 + exprs2, "RowNumber_CTE")
        self.mBackEnd.debrief_cte(self.mRowNumber_CTE)

    def addOriginalSignalAggreagtes(self, table, tr):
        lName = tr.mTransformation.get_name("");
        TS1 = alias(select([table.columns[self.mSignalName], table.columns[self.mRowNumberAlias]]), "t1");
        sig_expr = TS1.c[ self.mSignalName ]; # original signal
        rn_expr_1 = table.c[self.mRowNumberAlias];
        rn_expr_2 = TS1.c[self.mRowNumberAlias];
        cumulated_signal = null();
        previous_expr = null();
        relatve_diff = null();
        if(lName == ""):
            pass;
        elif (lName == "CumSum_"):           
            cumulated_signal  = select([func.sum(sig_expr)]).where(rn_expr_1 >= rn_expr_2);
        elif (lName == "Diff_"):
            previous_expr = select([func.sum(sig_expr)]).where(rn_expr_1 == (rn_expr_2 + 1));
            previous_expr = select([func.sum(sig_expr)]).where(rn_expr_1 == (rn_expr_2 + 1));
        elif (lName == "RelDiff_"):
            lMinExpr = self.mBackEnd.getFloatLiteral(tr.mTransformation.mMinValue);
            lDeltaExpr = self.mBackEnd.getFloatLiteral(tr.mTransformation.mDelta);
            lNormalizedSignal = (sig_expr - lMinExpr) / lDeltaExpr;
            lNormalizedLag1 = (previous_expr - lMinExpr) / lDeltaExpr;
            relatve_diff = (lNormalizedSignal - lNormalizedLag1) / (lNormalizedLag1 + 1.0);
        else:
            assert(0);

        cumulated_signal = cumulated_signal.label("Cum_" + self.mSignalName);
        previous_expr = previous_expr.label("Lag1_" + self.mSignalName);
        relatve_diff = relatve_diff.label("RelDiff_" + self.mSignalName);
        exprs = [cumulated_signal , previous_expr, relatve_diff];
        return exprs

    def generateTransformationInputCode(self, table):
        # => TransformationInput_CTE
        exprs1 = self.addOriginalSignalAggreagtes(self.mRowNumber_CTE,
                                                  self.mAutoForecast.mSignalDecomposition.mBestTransformation);
        self.mTransformationInputs_CTE = self.mBackEnd.generate_CTE(self.mRowNumber_CTE.columns  + exprs1, "TransformationInput_CTE")
        self.mBackEnd.debrief_cte(self.mTransformationInputs_CTE)

    def addTransformedSignal(self, table, tr):
        exprs = [];
        lName = tr.mTransformation.get_name("");
        # lName = "Diff_"
        TS1 = alias(select([table.columns[self.mSignalName], table.columns[self.mRowNumberAlias]]), "t1");
        sig_expr = TS1.c[ self.mSignalName ]; # original signal
        if(lName == ""):            
            trasformed_signal = table.c[ self.mSignalName ];
        elif(lName == "CumSum_"):
            trasformed_signal  = table.c["Cum_" + self.mSignalName];
        elif(lName == "Diff_"):
            lDefault_expr = self.mBackEnd.getFloatLiteral(tr.mTransformation.mFirstValue)
            previous_expr = func.coalesce(table.c["Lag1_" + self.mSignalName] , lDefault_expr);
            trasformed_signal  = table.c[ self.mSignalName ] - previous_expr;
        elif(lName == "RelDiff_"):
            lDefault_expr = self.mBackEnd.getFloatLiteral(tr.mTransformation.mFirstValue)
            lDefault_expr = (lDefault_expr - tr.mTransformation.mMinValue) / tr.mTransformation.mDelta;
            lRelDiff = func.coalesce(table.c["RelDiff_" + self.mSignalName] , lDefault_expr);
            trasformed_signal  = lRelDiff;
        else:
            assert(0);
            
        trasformed_signal = trasformed_signal.label(self.mSignalAlias);
        exprs = exprs + [ trasformed_signal ];
        return exprs

    def generateTransformationCode(self, table):
        # => Transformation_CTE
        signal_exprs = self.addTransformedSignal(self.mTransformationInputs_CTE,
                                                 self.mAutoForecast.mSignalDecomposition.mBestTransformation) 
        self.mTransformation_CTE = self.mBackEnd.generate_CTE([self.mTransformationInputs_CTE]  + signal_exprs, "Transformation_CTE")
        self.mBackEnd.debrief_cte(self.mTransformation_CTE)


    def addTrendInputs(self, table):
        tr = self.mAutoForecast.mSignalDecomposition.mBestTransformation;
        lName = tr.mTransformation.get_name("");
        exprs = [] ;
        normalized_time = table.c["NTime"];
        normalized_time_2 = normalized_time * normalized_time
        normalized_time_2 = normalized_time_2.label("NTime_2")
        normalized_time_3 = normalized_time_2 * normalized_time     
        normalized_time_3 = normalized_time_3.label("NTime_3")
        if(self.mTrend.mFormula == "Lag1Trend"):
            TS1 = alias(select([table.columns[self.mSignalAlias], table.columns[self.mRowNumberAlias]]), "t1");
            sig_expr = TS1.c[ self.mSignalAlias ];
            rn_expr_1 = table.c[self.mRowNumberAlias];
            rn_expr_2 = TS1.c[self.mRowNumberAlias];
            lag1 = select([func.sum(sig_expr)]).where(rn_expr_1 == (rn_expr_2 + 1));
            lag1 = lag1.label("Lag1")
            exprs = exprs + [lag1];
        exprs = exprs + [ normalized_time, normalized_time_2, normalized_time_3]
        return exprs
    
    def generateTrendInputCode(self):
        # => Trend_Inputs_CTE
        trend_inputs = self.addTrendInputs(self.mTransformation_CTE) 
        self.mTrend_inputs_CTE = self.mBackEnd.generate_CTE([self.mTransformation_CTE] + trend_inputs, "TICTE")
        self.mBackEnd.debrief_cte(self.mTrend_inputs_CTE)

    def generateTrendExpression(self, table):
        trend_expr = None;
        if(self.mTrend.mFormula == "ConstantTrend"):
            trend_expr = self.mBackEnd.getFloatLiteral(self.mTrend.mMean);
            pass
        elif(self.mTrend.mFormula == "LinearTrend"):
            print(self.mTrend.mTrendRidge.__dict__);
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"] + self.mBackEnd.getFloatLiteral(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.mFormula == "PolyTrend"):
            print(self.mTrend.mTrendRidge.__dict__);
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"];
            trend_expr += self.mTrend.mTrendRidge.coef_[1] * table.c["NTime_2"];
            trend_expr += self.mTrend.mTrendRidge.coef_[2] * table.c["NTime_3"];
            trend_expr += self.mBackEnd.getFloatLiteral(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.mFormula == "Lag1Trend"):
            lDefault_expr = self.mBackEnd.getFloatLiteral(self.mTrend.mMean);
            lag1 = func.coalesce(table.c["Lag1"] , lDefault_expr);
            trend_expr = lag1;
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
        trends = self.addTrends(self.mTrend_inputs_CTE)
        self.mTrend_CTE = self.mBackEnd.generate_CTE([self.mTrend_inputs_CTE] + trends, "TCTE")
        self.mBackEnd.debrief_cte(self.mTrend_CTE)



    def addCycleInputs(self, table):
        lTime =  self.mDateName;
        exprs = [];
        print(table.columns);
        lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            date_expr = table.c[lTime]
            date_parts = [extract('year', date_expr).label(lTime + "_Year") ,  
                          extract('month', date_expr).label(lTime + "_MonthOfYear") ,  
                          extract('day', date_expr).label(lTime + "_DayOfMonth") ,  
                          extract('hour', date_expr).label(lTime + "_Hour") ,  
                          extract('minute', date_expr).label(lTime + "_Minute") ,  
                          extract('second', date_expr).label(lTime + "_Second") ,  
                          func.weekday(date_expr).label(lTime + "_DayOfWeek") ,  
                          func.week(date_expr).label(lTime + "_WeekOfYear")
                          ]
            exprs = exprs + date_parts
        return exprs

    def generateCycleInputCode(self):
        # => Cycle_Inputs_CTE
        cycle_inputs = self.addCycleInputs(self.mTrend_CTE)
        self.mCycle_input_CTE = self.mBackEnd.generate_CTE([self.mTrend_CTE] + cycle_inputs, "CICTE")
        self.mBackEnd.debrief_cte(self.mCycle_input_CTE)

    
    def generateCaseWhen(self, iExpr , iValue , iOutput, iElse):
        cond1 = None;
        cond1 = (iExpr == iValue);
        expr1 = case([(cond1 , iOutput)], else_ = iElse);
        return expr1;


    def generateCycleSpecificExpression(self, iExpr, iDict, iDefault):
        expr_2 = None;
        lDict = iDict;
        key, value = lDict.popitem()
        expr_1 = None;
        if(len(lDict) > 0):
            expr_1 = self.generateCycleSpecificExpression(iExpr, lDict, iDefault);
        else:
            return self.mBackEnd.getFloatLiteral(iDefault);
        valueexpr =  self.mBackEnd.getFloatLiteral(value);
        expr_2 = self.generateCaseWhen(iExpr, int(key), valueexpr, expr_1);
        return expr_2;


    def generateCycleExpression(self, table):
        cycle_expr = None;
        if(self.mCycle.mFormula == "NoCycle"):
            cycle_expr = self.mBackEnd.getFloatLiteral(0.0);
            pass
        elif(self.mCycle.mFormula.startswith("Seasonal_")):
            lExpr = table.c[self.mDateName + "_" + self.mCycle.mDatePart]
            cycle_expr = self.generateCycleSpecificExpression(lExpr ,
                                                              self.mCycle.mEncodedValueDict,
                                                              self.mCycle.mDefaultValue);
            pass
        elif(self.mCycle.mFormula.startswith("Cycle_None")):
            cycle_expr = self.mBackEnd.getFloatLiteral(0.0);
            pass
        elif(self.mCycle.mFormula.startswith("Cycle_")):
            lExpr = table.c[self.mRowNumberAlias]
            lExpr = lExpr % int(self.mCycle.mBestCycleLength)
            cycle_expr = self.generateCycleSpecificExpression(lExpr ,
                                                              self.mCycle.mBestCycleValueDict,
                                                              self.mCycle.mDefaultValue);
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
        self.mCycle_CTE = self.mBackEnd.generate_CTE([self.mCycle_input_CTE] + cycles, "CYCTE")
        self.mBackEnd.debrief_cte(self.mCycle_CTE)


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
        cycle_resdiues = self.addCycleResidues(self.mCycle_CTE)
        self.mCycle_residues_CTE = self.mBackEnd.generate_CTE([self.mCycle_CTE] + cycle_resdiues, "CYRESCTE")
        self.mBackEnd.debrief_cte(self.mCycle_residues_CTE)


    def createLags(self, table , H , col, index_col):
        # TS0 = table;
        TS1 = None;
        if(self.mBackEnd.supports_CTE()):
            TS1 = alias(select([table.c[index_col] , table.c[col]]), "KKKKKK");
        else:
            TS1 = alias(select([table.c[index_col] , table.c[col]]), "KKKKKK");
        col_expr_1 = TS1.c[col];
        index_expr = table.c[index_col];
        index_expr_1 = TS1.c[index_col];
        exprs = [];
        for h1 in range(1 , H+1):
            case1 = case([(index_expr == (index_expr_1 + h1) , col_expr_1)] , else_ = null());
            expr = select([func.sum(case1)]).select_from(table);
            expr = expr.label(col + "_Lag" + str(h1));
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
        ar_inputs = self.addARInputs(self.mCycle_residues_CTE)
        self.mAR_input_CTE = self.mBackEnd.generate_CTE([self.mCycle_residues_CTE] + ar_inputs, "ARICTE")
        self.mBackEnd.debrief_cte(self.mAR_input_CTE)
        

    def addARModel(self, table):
        exprs = [];
        ar_expr = None;
        if(self.mAR.mFormula != "NoAR"):
            lDefault_expr = self.mBackEnd.getFloatLiteral(self.mAR.mDefaultValue);
            i = 0 ;
            for i in range(len(self.mAR.mARLagNames)):
                feat = self.Shortened[self.mCycle.getCycleResidueName()] + "_Lag" + str(i+1);
                feat_value = func.coalesce(table.c[feat] , lDefault_expr);
                if(ar_expr is None):
                    ar_expr = self.mAR.mARRidge.coef_[i] * feat_value;
                else:
                    ar_expr = ar_expr + self.mAR.mARRidge.coef_[i] * feat_value;
                i = i + 1;
            ar_expr = ar_expr + self.mBackEnd.getFloatLiteral(self.mAR.mARRidge.intercept_);
        else:
            ar_expr = self.mBackEnd.getFloatLiteral(0.0);
        ar_expr = ar_expr.label(self.Shortened[self.mAR.mOutName])
        exprs = exprs + [ar_expr]
        return exprs

    def generateARCode(self):
        # => AR_CTE
        ars = self.addARModel(self.mAR_input_CTE)
        self.mAR_CTE = self.mBackEnd.generate_CTE([self.mAR_input_CTE] + ars, "ARCTE")
        self.mBackEnd.debrief_cte(self.mAR_CTE)

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
        self.mModel_CTE = self.mBackEnd.generate_CTE([self.mAR_CTE] + model_vars, "MODCTE")
        self.mBackEnd.debrief_cte(self.mModel_CTE)

