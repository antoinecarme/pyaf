# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

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
from sqlalchemy.sql import expression

import pandas as pd;
import numpy as np;

from dateutil.tz import tzutc

import pyaf.TS 

@compiles(DropTable, "postgresql")
def _compile_drop_table(element, compiler, **kwargs):
    return compiler.visit_drop_table(element) + " CASCADE"

class date_diff(GenericFunction):
    type = Float
    name = 'date_diff'
    
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
def _pg_date_diff(element, compiler, **kw):  # pragma: no cover
    arg1, arg2 = list(element.clauses)
    return "(extract(epoch from %s) - extract(epoch from %s))" % (compiler.process(arg1),
                                                                  compiler.process(arg2));

@compiles(date_diff, 'sqlite')
def _sl_date_diff(element, compiler, **kw):    # pragma: no cover
    return "julianday(%s) - julianday(%s)" % (compiler.process(element.clauses.clauses[0]),
                                              compiler.process(element.clauses.clauses[1]),
                                              )

class date_add(GenericFunction):
    type = DateTime
    name = 'date_add'
    
@compiles(date_add, 'default')
def _default_date_add(element, compiler, **kw):  # pragma: no cover
    return "DATEDIFF(%s, %s)" % (compiler.process(element.clauses.clauses[0]),
                                 compiler.process(element.clauses.clauses[1]),
                                 )
@compiles(date_add, 'mysql')
def _my_date_add(element, compiler, **kw):  # pragma: no cover
    return "DATE_ADD(%s, INTERVAL %s SECOND)" % (compiler.process(element.clauses.clauses[0]),
                                 compiler.process(element.clauses.clauses[1]),
                                 )

@compiles(date_add, 'postgresql')
def _pg_date_add(element, compiler, **kw):  # pragma: no cover
    arg1, arg2 = list(element.clauses)
    return "(%s + (%s) * INTERVAL '1 SECOND')" % (compiler.process(element.clauses.clauses[0]),
                                                  compiler.process(element.clauses.clauses[1]));

@compiles(date_add, 'sqlite')
def _sl_date_add(element, compiler, **kw):    # pragma: no cover
    return "datetime(%s, '+%s seconds')" % (compiler.process(element.clauses.clauses[0]),
                                              compiler.process(element.clauses.clauses[1]),
                                              )

class weekday(GenericFunction):
    type = Float

@compiles(weekday)
def _def_weekday(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(dow from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(weekday, 'mysql')
def _my_weekday(element, compiler, **kw):  # pragma: no cover
    return "WEEKDAY(%s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(weekday, 'postgresql')
def _pg_weekday(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(dow from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(weekday, 'sqlite')
def _sl_weekday(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(dow, %s)" % (compiler.process(element.clauses.clauses[0]))

class week(GenericFunction):
    type = Float

@compiles(week)
def _def_week(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(week from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(week, 'mysql')
def _my_week(element, compiler, **kw):  # pragma: no cover
    return "WEEK(%s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(week, 'postgresql')
def _pg_week(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(week from %s)" % (compiler.process(element.clauses.clauses[0]))

@compiles(week, 'sqlite')
def _sl_week(element, compiler, **kw):  # pragma: no cover
    return "EXTRACT(week, %s)" % (compiler.process(element.clauses.clauses[0]))


class cDatabaseBackend:
    
    def __init__(self, iDSN = None, iDialect = None):
        self.mDebug = False;
        
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

        self.mShortLabels = {};
        pass

    def addShortLabel(self, iLongName, iShortName):
        self.mShortLabels[iLongName] = iShortName;

    def initializeEngine(self):
        if(self.mDSN is not None):
            # connected mode.
            self.mEngine = create_engine(self.mDSN , echo = True)
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
        # print("GENERATE_SQL_STATEMENT" , statement);
        lResult = statement.compile(bind=self.mEngine, compile_kwargs={'literal_binds': True});
        print("GENERATE_SQL_RESULT" , lResult.string[0:200]);
        return lResult.string;
    
    def createLogicalTable(self, iTableName, iDateName, iSignalName, iDateType, iExogenousVariables = []):
        lTestTableName = iTableName;
        lPrimaryKeyName = iDateName;
        
        lTestTable = Table(lTestTableName,
                           self.mMeta);

        lTestTable.append_column(Column(iDateName, iDateType));
        lTestTable.append_column(Column(iSignalName, Float));
        for exog in iExogenousVariables:
            lTestTable.append_column(Column(exog, String));            
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
            # print("MYSQL_DATABASE_SUPPORTS_CTE_AND_ROW_NUMBER" , str(lMajor) +"." + str(lMinor));
            return True;
        return False;

    def supports_CTE(self):
        #return False;
        lDialectName = self.getDialectName();
        # if(("sqlite" == lDialectName) or (("mysql" == lDialectName) and not self.isNewMySQL())):
        #     return False;
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
        lGeneratedApplyOut.columns = result.keys();
        return lGeneratedApplyOut;

    def generateRandomTableName(self, length = 8):
        #        return "test_table";
        chars = string.ascii_uppercase + string.digits;
        lPrefix = "TS_CODEGEN_";
        lRandomChars = ''.join(random.choice(chars) for _ in range(length))
        return lPrefix + lRandomChars;


    def materializeTable(self, iDataFrame, iTableName):
        lTestTableName = iTableName;
        iDataFrame.to_sql(lTestTableName , self.mConnection,  if_exists='replace', index=False)
        lTestTable = Table(lTestTableName,
                           self.mMeta,
                           autoload=True,
                           autoload_with = self.mEngine);
        return lTestTable;
        
        
    def getIntegerLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.Integer);

    def getFloatLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.FLOAT);

    def getDateTimeLiteral(self, iValue):
        lDialectName = self.getDialectName();

        # return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.TIMESTAMP);
        if("sqlite" == lDialectName):
            return sqlalchemy.sql.expression.literal_column("'" + str(iValue) + "'", sqlalchemy.types.DATETIME);
        # return func.cast(str(iValue), sqlalchemy.types.TIMESTAMP);
        return sqlalchemy.sql.expression.literal_column("TIMESTAMP '" + str(iValue) + "'", sqlalchemy.types.DATETIME); # .strftime("%Y-%m-%d %H:%M:%S")

    def getIntervalLiteral(self, iValue, h):
        return sqlalchemy.sql.expression.literal_column(str(h) + " * INTERVAL '" + str(iValue) + "'", sqlalchemy.types.Interval);
    
    def debrief_cte(self, cte):
        if(not self.mDebug):
            return;
        print("debrief_cte_cte" , cte);
        print("CTE_COLUMNS" , cte.columns);
        print("CTE_COLUMNS_TYPES" , [col.type for col in cte.columns]);
        statement = select(cte.columns);
        print("debrief_cte_statement" , statement);
        lSQL = self.generate_Sql(statement);
        print("********************" , self.getDialectName() , "***********************************");
        print(lSQL[0:200]);
        if(self.isConnected()):
            lGeneratedApplyOut = self.executeSQL(lSQL);
            print(lGeneratedApplyOut.info());
            print(lGeneratedApplyOut.head());
            print(lGeneratedApplyOut.tail());
        print("********************" , self.getDialectName() , "***********************************");
        return lSQL;

    def generateSQLForStatement(self, cte):
        statement = select([cte] , use_labels=True)
        # statement = alias(select([cte]), "SQLGenResult");
        lSQL = self.generate_Sql(statement);
        return lSQL;


'''
************************************************************************************************************************



***********************************************************************************************************************88

'''

class cDecompositionCodeGenObject:
    def __init__(self, iDSN = None, iDialect = None):
        sys.setrecursionlimit(10000)
        self.mForecastEngine = None;
        self.mBackEnd = cDatabaseBackend(iDSN , iDialect);
    
    def generateCode(self, iAutoForecast, iTableName = None):
        lTableName = iTableName;
        if(lTableName is None):
            lTableName = self.mBackEnd.generateRandomTableName();
        self.mForecastEngine = iAutoForecast;
        self.generateCode_Internal(lTableName);
        lSQL = self.mBackEnd.generateSQLForStatement(self.mFinal_CTE);
        return lSQL;

    def testGeneration(self, iAutoForecast, iTableName = None):
        df = iAutoForecast.mSignalDecomposition.mTrainingDataset;
        lTableName = iTableName;
        if(lTableName is None):
            lTableName = self.mBackEnd.generateRandomTableName();
        lTestTable = self.mBackEnd.materializeTable(df, lTableName);
        lSQL = self.generateCode(iAutoForecast, lTableName);
        H = self.getHorizon();        
        lInternalApplyOut = iAutoForecast.forecast(df, H);
        print(lInternalApplyOut.info());
        # print(lInternalApplyOut.head(2*H));
        print(lInternalApplyOut.tail(2*H));                
        lGeneratedApplyOut = self.mBackEnd.executeSQL(lSQL);
        lTestTable.drop();
        select1 = select([self.mFinal_CTE]);
        lGeneratedApplyOut.columns = select1.columns.keys();
        print(lGeneratedApplyOut.info());
        # print(lGeneratedApplyOut.head(2*H));
        print(lGeneratedApplyOut.tail(2*H));        
        # lGeneratedApplyOut.to_csv("sql_generated.csv");
        

    def shorten(self, iName):
        if(iName in self.mBackEnd.mShortLabels.keys()):
            return self.mBackEnd.mShortLabels[iName];
        lLength = len(iName);
        if(lLength < 30):
            return iName;
        chars = string.ascii_uppercase + string.digits;
        lPrefix = "AutoLabel_";
        lRandomChars = ''.join(random.choice(chars) for _ in range(6));
        lName = iName[(lLength - 15) : ];
        lShort = lPrefix + lRandomChars + "_" + lName;
        self.mBackEnd.addShortLabel(iName, lShort);
        return lShort; 

    def getHorizon(self):
        return self.mTimeInfo.mHorizon;

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

         self.mBestModel = self.mForecastEngine.mSignalDecomposition.mBestModel;
         self.mTimeInfo = self.mBestModel.mTimeInfo

         self.mTrend = self.mBestModel.mTrend
         self.mCycle =  self.mBestModel.mCycle
         self.mAR =  self.mBestModel.mAR

         self.mExogenousInfo = self.mAR.mExogenousInfo;

         self.mOriginalSignal = self.mBestModel.mOriginalSignal
         self.mDateName = self.mBestModel.mTime
         self.mModelName =  self.mBestModel.mOutName
         print(self.mBestModel.mTransformation.__class__);
         lNeedTransformation = (self.mBestModel.mTransformation.__class__ != pyaf.TS.Signal_Transformation.cSignalTransform_None);

         self.mRowTypeAlias = "HType" 
         self.mSignal = self.mBestModel.mSignal; 
         self.mRowNumberAlias = "RN";

         self.mDateType = sqlalchemy.types.FLOAT
         lTimeInfo = self.mBestModel.mTimeInfo
         if(lTimeInfo.isPhysicalTime()):
             self.mDateType = sqlalchemy.types.TIMESTAMP;

         self.mBackEnd.addShortLabel(self.mTrend.mOutName, "Trend");
         self.mBackEnd.addShortLabel(self.mCycle.mOutName, "Cycle");
         self.mBackEnd.addShortLabel(self.mAR.mOutName, "AR");
         self.mBackEnd.addShortLabel(self.mCycle.getCycleName(), "Cycle");
         self.mBackEnd.addShortLabel(self.mCycle.getCycleResidueName(), "CycleRes");
         # self.mBackEnd.addShortLabel(self.mModelName, "Model");
         # print(self.Shortened);

         lTableName = iTableName;
         if(lTableName is None):
             lTableName = self.mBackEnd.generateRandomTableName();
         base_table = self.mBackEnd.createLogicalTable(lTableName , 
                                                       self.mDateName,
                                                       self.mOriginalSignal, self.mDateType,
                                                       self.getExogenousVariables());
         table = base_table;

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
         if(lNeedTransformation):
             self.generateModelInvTransformationInputCode();
             self.generateInverseTransformationCode(); # => Model_CTE
             self.mFinal_CTE = self.mModelInvTransformation_CTE;
         else:
             self.mFinal_CTE = self.mModel_CTE;
         self.generateForecasts();
         self.mFinal_CTE = self.mForecast_CTE;


    def getExogenousVariables(self):
        lExogenousVariables = [];
        lExogenousInfo = self.mExogenousInfo;
        if(lExogenousInfo is not None):
            lExogenousVariables = lExogenousInfo.mExogenousVariables;
        return lExogenousVariables;

    def addNormalizedTime(self, table):
        exprs = [];
        normalized_time = None;
        lTimeInfo = self.mBestModel.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            lType = sqlalchemy.types.TIMESTAMP;
            lMinExpr = self.mBackEnd.getDateTimeLiteral(lTimeInfo.mTimeMin);
            lMaxExpr = self.mBackEnd.getDateTimeLiteral(lTimeInfo.mTimeMax)
            # exprs = exprs + [lMinExpr, lMaxExpr];
            # count the number of days here if date. number of seconds if datetime
            lAmplitude = func.date_diff(lMaxExpr, lMinExpr);
            lAmplitude = func.cast(lAmplitude, sqlalchemy.types.FLOAT);
            normalized_time = func.date_diff(table.c[self.mDateName],  lMinExpr);
            normalized_time = func.cast(normalized_time, sqlalchemy.types.FLOAT);
        else:
            lAmplitude = self.as_float(lTimeInfo.mTimeMax - lTimeInfo.mTimeMin)
            normalized_time = table.c[self.mDateName] - self.as_float(lTimeInfo.mTimeMin);
        normalized_time = normalized_time / lAmplitude
        normalized_time = normalized_time.label("NTime")
        exprs = exprs + [normalized_time];
        return exprs
        
    def addRowNumber_analytical(self, table):
        exprs = [];
        row_number_column = func.row_number().over(order_by=asc(table.c[self.mDateName]))
        row_number_column = func.cast(row_number_column, sqlalchemy.types.Integer);
        row_number_column = row_number_column.label(self.mRowNumberAlias)
        exprs = exprs + [ row_number_column];
        return exprs

    def addRowNumber_as_count(self, table):
        exprs = [];
        TS1 = alias(select([table.columns[self.mDateName]]), "TS_CTE_RowNum");
        time_expr_1 = table.c[self.mDateName];
        time_expr_2 = TS1.c[self.mDateName];
        
        expr = select([func.count(time_expr_2)]).where(time_expr_1 > time_expr_2);
        row_number_column = expr;
        row_number_column = row_number_column.label(self.mRowNumberAlias)
        exprs = exprs + [ row_number_column];
        return exprs

    def addExogenousColumns(self, table):
        exprs = [];
        lExogenousVariables = self.getExogenousVariables();
        for exog in lExogenousVariables:
            exprs.append(table.c[exog]);
        return exprs;

    def generateRowNumberCode(self, table):
        # => RowNumber_CTE
        exprs1 = None;
        if(self.mBackEnd.hasAnalyticalRowNumber()):
            exprs1 = self.addRowNumber_analytical(table);
        else:
            exprs1 = self.addRowNumber_as_count(table);
        exprs2 = self.addNormalizedTime(table);
        exprs3 = self.addExogenousColumns(table);

        lMainColumns = [table.c[self.mDateName] , table.c[self.mOriginalSignal] ]
        
        self.mRowNumber_CTE = self.mBackEnd.generate_CTE(lMainColumns  + exprs1 + exprs2 + exprs3, "TS_CTE_RowNumber")
        # self.mBackEnd.debrief_cte(self.mRowNumber_CTE)

    def addOriginalSignalAggreagtes(self, table):
        lTrClass = self.mBestModel.mTransformation.__class__;
        TS1 = alias(select([table.columns[self.mOriginalSignal], table.columns[self.mRowNumberAlias]]), "t_SigRowNum");
        sig_expr = TS1.c[ self.mOriginalSignal ]; # original signal
        rn_expr_1 = table.c[self.mRowNumberAlias];
        rn_expr_2 = TS1.c[self.mRowNumberAlias];
        cumulated_signal = None;
        previous_expr = None;
        relatve_diff = None;
        if(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_None):
            pass;
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Accumulate):           
            cumulated_signal  = select([func.sum(sig_expr)]).where(rn_expr_1 >= rn_expr_2);
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Differencing):
            previous_expr = select([func.sum(sig_expr)]).where(rn_expr_1 == (rn_expr_2 + 1));
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_RelativeDifferencing):
            lMinExpr = self.as_float(tr.mTransformation.mMinValue);
            lDeltaExpr = self.as_float(tr.mTransformation.mDelta);
            lNormalizedSignal = (sig_expr - lMinExpr) / lDeltaExpr;
            lNormalizedLag1 = (previous_expr - lMinExpr) / lDeltaExpr;
            relatve_diff = (lNormalizedSignal - lNormalizedLag1) / (lNormalizedLag1 + 1.0);
        else:
            assert(0);

        exprs = [];
        if(cumulated_signal is not None):
            cumulated_signal = cumulated_signal.label("Cum_" + self.mOriginalSignal);
            exprs = [cumulated_signal] + exprs;
            
        if(previous_expr is not None):
            previous_expr = previous_expr.label("Lag1_" + self.mOriginalSignal);
            exprs = [previous_expr] + exprs;
        
        if(relatve_diff is not None):
            relatve_diff = relatve_diff.label("RelDiff_" + self.mOriginalSignal);
            exprs = [relatve_diff] + exprs;

        return exprs

    def addExogenousDummies(self, table):
        exprs = [];
        lExogenousInfo = self.mExogenousInfo;
        if(lExogenousInfo is None):
            return exprs;

        lExogenousVariables = self.getExogenousVariables();

        print(table.columns);
        for exog in lExogenousVariables:
            lList = lExogenousInfo.mExogenousVariableCategories[exog];
            if(lList is not None):
                for lCat in lList:
                    lDummyName = exog + "=" + str(lCat);
                    lCatExpr = sqlalchemy.sql.expression.literal(str(lCat), String);
                    if((lCat == "") or (lCat is None)):                    
                        cond = (table.c[exog] == None);
                    else:
                        cond = (table.c[exog] == lCatExpr);
                    expr1 = case([(cond , self.as_float(1.0))],
                                 else_ = self.as_float(0.0));
                    expr1 = expr1.label(lDummyName);
                    exprs = exprs + [expr1];
            else:                
                lMeanExpr = self.as_float(lExogenousInfo.mContExogenousStats[exog][0]);
                lStdExpr = self.as_float(lExogenousInfo.mContExogenousStats[exog][1]);
                lEncodedContExog = (table.c[exog] - lMeanExpr) / lStdExpr;
                exprs = exprs + [lEncodedContExog];
        return exprs;

    def generateTransformationInputCode(self, table):
        # => TransformationInput_CTE
        exprs1 = self.addOriginalSignalAggreagtes(self.mRowNumber_CTE);
        exprs2 = self.addExogenousDummies(self.mRowNumber_CTE);
        self.mTransformationInputs_CTE = self.mBackEnd.generate_CTE(self.mRowNumber_CTE.columns  + exprs1 + exprs2, "TS_CTE_TransformationInput")
        # self.mBackEnd.debrief_cte(self.mTransformationInputs_CTE)

    def addTransformedSignal(self, table):
        exprs = [];
        lTrClass = self.mBestModel.mTransformation.__class__;
        # lName = "Diff_"
        TS1 = alias(select([table.columns[self.mOriginalSignal], table.columns[self.mRowNumberAlias]]), "t_SigRowNum2");
        sig_expr = TS1.c[ self.mOriginalSignal ]; # original signal
        if(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_None):            
            trasformed_signal = table.c[ self.mOriginalSignal ];
        elif(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Accumulate):
            trasformed_signal  = table.c["Cum_" + self.mOriginalSignal];
        elif(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Differencing):
            lDefault_expr = self.as_float(tr.mTransformation.mFirstValue)
            previous_expr = func.coalesce(table.c["Lag1_" + self.mOriginalSignal] , lDefault_expr);
            trasformed_signal  = table.c[ self.mOriginalSignal ] - previous_expr;
        elif(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_RelativeDifferencing):
            lDefault_expr = self.as_float(tr.mTransformation.mFirstValue)
            lDefault_expr = (lDefault_expr - tr.mTransformation.mMinValue) / tr.mTransformation.mDelta;
            lRelDiff = func.coalesce(table.c["RelDiff_" + self.mOriginalSignal] , lDefault_expr);
            trasformed_signal  = lRelDiff;
        else:
            assert(0);
            
        trasformed_signal = trasformed_signal.label(self.mSignal);
        exprs = exprs + [ trasformed_signal ];
        return exprs

    def generateTransformationCode(self, table):
        # => Transformation_CTE
        signal_exprs = self.addTransformedSignal(self.mTransformationInputs_CTE) 
        self.mTransformation_CTE = self.mBackEnd.generate_CTE([self.mTransformationInputs_CTE]  + signal_exprs, "TS_CTE_Transformation")
        # self.mBackEnd.debrief_cte(self.mTransformation_CTE)


    def addTrendInputs(self, table):
        exprs = [] ;
        normalized_time = table.c["NTime"];
        normalized_time_2 = normalized_time * normalized_time
        normalized_time_2 = normalized_time_2.label("NTime_2")
        normalized_time_3 = normalized_time_2 * normalized_time     
        normalized_time_3 = normalized_time_3.label("NTime_3")
        if(self.mTrend.__class__ == pyaf.TS.SignalDecomposition_Trend.cLag1Trend):
            TS1 = alias(select([table.columns[self.mSignal], table.columns[self.mRowNumberAlias]]), "t_SigRowNum3");
            sig_expr = TS1.c[ self.mSignal ];
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
        self.mTrend_inputs_CTE = self.mBackEnd.generate_CTE([self.mTransformation_CTE] + trend_inputs, "TS_CTE_Trend_Input")
        # self.mBackEnd.debrief_cte(self.mTrend_inputs_CTE)

    def as_float(self, x):
        return self.mBackEnd.getFloatLiteral(float(x));

    def generateTrendExpression(self, table):
        print("TREND" , self.mTrend.__class__, self.mTrend.mFormula)
        trend_expr = None;
        if(self.mTrend.__class__ == pyaf.TS.SignalDecomposition_Trend.cConstantTrend):
            trend_expr = self.as_float(self.mTrend.mMean);
            pass
        elif(self.mTrend.__class__ == pyaf.TS.SignalDecomposition_Trend.cLinearTrend):
            # print(self.mTrend.mTrendRidge.__dict__);
            lCoeffs = self.mTrend.mTrendRidge.coef_;
            trend_expr = self.as_float(lCoeffs[0]) * table.c["NTime"] + self.as_float(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.__class__ == pyaf.TS.SignalDecomposition_Trend.cPolyTrend):
            # print(self.mTrend.mTrendRidge.__dict__);
            lCoeffs = self.mTrend.mTrendRidge.coef_;
            trend_expr = self.as_float(lCoeffs[0]) * table.c["NTime"];
            trend_expr += self.as_float(lCoeffs[1]) * table.c["NTime_2"];
            trend_expr += self.as_float(lCoeffs[2]) * table.c["NTime_3"];
            trend_expr += self.as_float(self.mTrend.mTrendRidge.intercept_)
            pass
        elif(self.mTrend.__class__ == pyaf.TS.SignalDecomposition_Trend.cLag1Trend):
            lDefault_expr = self.as_float(self.mTrend.mDefaultValue);
            lag1 = func.coalesce(table.c["Lag1"] , lDefault_expr);
            trend_expr = lag1;
            pass
        return trend_expr;


    def addTrends(self, table):
        exprs = [];
        trend_expr = self.generateTrendExpression(table);
        #print(type(trend_expr))
        
        trend_expr = trend_expr.label(self.shorten(self.mTrend.mOutName));
        exprs = exprs + [trend_expr]
        return exprs

    def generateTrendCode(self):
        # => Trend_CTE
        trends = self.addTrends(self.mTrend_inputs_CTE)
        self.mTrend_CTE = self.mBackEnd.generate_CTE([self.mTrend_inputs_CTE] + trends, "TS_CTE_Trend")
        # self.mBackEnd.debrief_cte(self.mTrend_CTE)



    def addCycleInputs(self, table):
        lTime =  self.mDateName;
        exprs = [];
        # print(table.columns);
        if(self.mCycle.__class__ == pyaf.TS.SignalDecomposition_Cycle.cSeasonalPeriodic):
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
        self.mCycle_input_CTE = self.mBackEnd.generate_CTE([self.mTrend_CTE] + cycle_inputs, "TS_CTE_CylceInputs")
        # self.mBackEnd.debrief_cte(self.mCycle_input_CTE)

    
    def generateCaseWhen(self, iExpr , iValue , iOutput, iElse):
        cond1 = None;
        cond1 = (iExpr == iValue);
        expr1 = case([(cond1 , iOutput)], else_ = iElse);
        return expr1;


    def generateCycleSpecificExpression(self, iExpr, iDict, iDefault):
        if(len(iDict) == 0):
            return self.as_float(iDefault);
        lDict = iDict;
        key, value = lDict.popitem()
        expr_1 = self.generateCycleSpecificExpression(iExpr, lDict, iDefault);
        valueexpr =  self.as_float(value);
        expr_2 = self.generateCaseWhen(iExpr, self.mBackEnd.getIntegerLiteral(key), valueexpr, expr_1);
        return expr_2;


    def generateCycleExpression(self, table):
        cycle_expr = None;
        print("CYCLE" , self.mCycle.__class__, self.mCycle.mFormula)
        if(self.mCycle.__class__ == pyaf.TS.SignalDecomposition_Cycle.cZeroCycle):
            cycle_expr = self.as_float(0.0);
            pass
        elif(self.mCycle.__class__ == pyaf.TS.SignalDecomposition_Cycle.cSeasonalPeriodic):
            lExpr = table.c[self.mDateName + "_" + self.mCycle.mDatePart]
            cycle_expr = self.generateCycleSpecificExpression(lExpr ,
                                                              self.mCycle.mEncodedValueDict,
                                                              self.mCycle.mDefaultValue);
            pass
        elif(self.mCycle.__class__ == pyaf.TS.SignalDecomposition_Cycle.cBestCycleForTrend):
            lExpr = table.c[self.mRowNumberAlias] - 1
            lExpr = func.mod(lExpr, self.mBackEnd.getIntegerLiteral(int(self.mCycle.mBestCycleLength)))
            cycle_expr = self.generateCycleSpecificExpression(lExpr ,
                                                              self.mCycle.mBestCycleValueDict[self.mCycle.mBestCycleLength],
                                                              self.mCycle.mDefaultValue);
            pass
        return cycle_expr;
    


    def addCycles(self, table):
        exprs = [];
        cycle_expr = self.generateCycleExpression(table);
        cycle_expr = cycle_expr.label(self.shorten(self.mCycle.getCycleName()))
        exprs = exprs + [cycle_expr]
        return exprs
    
    def generateCycleCode(self):
        # => Cycle_CTE
        # sel1 = alias(select([]), "CIN")
        # print(sel1.columns)
        cycles = self.addCycles(self.mCycle_input_CTE)
        self.mCycle_CTE = self.mBackEnd.generate_CTE([self.mCycle_input_CTE] + cycles, "TS_CTE_Cycle")
        # self.mBackEnd.debrief_cte(self.mCycle_CTE)    

    def addCycleResidues(self, table):
        exprs = [];
        cycle_expr = table.c[self.shorten(self.mCycle.getCycleName())];
        trend_expr = table.c[self.shorten(self.mTrend.mOutName)];
        cycle_residue_expr = trend_expr + cycle_expr - table.c[self.mSignal]
        cycle_residue_expr = cycle_residue_expr.label(self.shorten(self.mCycle.getCycleResidueName()))
        exprs = exprs + [cycle_residue_expr]
        return exprs


    def generateCycleResidueCode(self):
        # => Cycle_Residue_CTE
        cycle_resdiues = self.addCycleResidues(self.mCycle_CTE)
        self.mCycle_residues_CTE = self.mBackEnd.generate_CTE([self.mCycle_CTE] + cycle_resdiues, "TS_CTE_CycleResidue")
        # self.mBackEnd.debrief_cte(self.mCycle_residues_CTE)


    def createLags(self, table , P , cols, index_col):
        # TS0 = table;
        index_expr = table.c[index_col];
        exprs = [];
        self.mDefaultARLagValues = {};
        # lCycRes = self.shorten(self.mCycle.getCycleResidueName());
        # self.mDefaultARLagValues[lCycRes] = self.mAR.getDefaultValue(self.mCycle.getCycleResidueName());
        for p in range(1 , P+1):
            for col in cols:
                lColumnVars = [ table.c[index_col] ];
                lColumnVars = lColumnVars + [ table.c[ self.shorten(col) ] ];
                TS1 = None;
                if(self.mBackEnd.supports_CTE()):
                    TS1 = alias(select(lColumnVars), "LagInput");
                else:
                    TS1 = alias(select(lColumnVars), "LagInput");
                col_expr_1 = TS1.c[ self.shorten(col) ];
                index_expr_1 = TS1.c[index_col];                
                case1 = case([(index_expr == (index_expr_1 + p) , col_expr_1)] , else_ = null());
                expr = select([func.sum(case1)]).select_from(table);
                lLong = col + "_Lag" + str(p);
                lLabel = self.shorten(col) + "_Lag" + str(p);
                self.mDefaultARLagValues[lLabel] = self.mAR.getDefaultValue(col);
                expr = expr.label(lLabel);
                exprs = exprs + [expr];
        return exprs;

    def addARInputs(self, table):
        lExogenousInfo = self.mExogenousInfo;
        exprs = [];
        if(self.mAR.mFormula != "NoAR"):            
            lVars = [ self.mCycle.getCycleResidueName() ];
            if(lExogenousInfo is not None):
                for exog in lExogenousInfo.mEncodedExogenous:
                    lVars = lVars + [exog];
            exprs = exprs + self.createLags(table, 
                                            self.mAR.mNbLags, 
                                            lVars,
                                            self.mRowNumberAlias);
        return exprs


    def generateARInputCode(self):
        # => AR_Inputs_CTE
        self.mAR_inputs = self.addARInputs(self.mCycle_residues_CTE)
        self.mAR_input_CTE = self.mBackEnd.generate_CTE([self.mCycle_residues_CTE] + self.mAR_inputs, "TS_CTE_AR_Inputs")
        # self.mBackEnd.debrief_cte(self.mAR_input_CTE)
        

    def addARModel(self, table):
        print("ARX" , self.mAR.__class__, self.mAR.mFormula)
        exprs = [];
        ar_expr = None;
        if(self.mAR.__class__ != pyaf.TS.SignalDecomposition_AR.cZeroAR):
            i = 0 ;
            for i in range(len(self.mAR_inputs)):
                lag = self.mAR_inputs[i];
                feat = lag.key;
                lDefault_expr = self.as_float(self.mDefaultARLagValues[feat]);
                feat_value = func.coalesce(table.c[feat] , lDefault_expr);
                if(ar_expr is None):
                    ar_expr = self.mAR.mScikitModel.coef_[i] * feat_value;
                else:
                    ar_expr = ar_expr + self.mAR.mScikitModel.coef_[i] * feat_value;
                i = i + 1;
            ar_expr = ar_expr + self.as_float(self.mAR.mScikitModel.intercept_);
        else:
            ar_expr = self.as_float(0.0);
        ar_expr = ar_expr.label(self.shorten(self.mAR.mOutName))
        exprs = exprs + [ar_expr]
        return exprs

    def generateARCode(self):
        # => AR_CTE
        ars = self.addARModel(self.mAR_input_CTE)
        self.mAR_CTE = self.mBackEnd.generate_CTE([self.mAR_input_CTE] + ars, "TS_CTE_AR")
        # self.mBackEnd.debrief_cte(self.mAR_CTE)

    def add_TS_Model(self, table):
        exprs = [];
        sum_1 = table.c[self.shorten(self.mTrend.mOutName)];
        sum_1 += table.c[self.shorten(self.mCycle.mOutName)];
        sum_1 += table.c[self.shorten(self.mAR.mOutName)];
        model_expr = sum_1;
        model_expr = model_expr.label(self.shorten("Model"))
        model_residue = model_expr - table.c[self.mSignal]
        model_residue = model_residue.label("Model" + "Residue")
        signal_or_forecast_expr = func.coalesce(table.c[self.mSignal], model_expr);
        signal_or_forecast_expr = signal_or_forecast_expr.label("PredictedSignal")
        exprs = exprs + [model_expr , model_residue, signal_or_forecast_expr]
        return exprs

    def generateModelCode(self):
        # => Model_CTE
        # sel1 = alias(select([]), "AR")
        # print(sel1.columns)
        model_vars = self.add_TS_Model(self.mAR_CTE)
        self.mModel_CTE = self.mBackEnd.generate_CTE([self.mAR_CTE] + model_vars, "TS_CTE_Model")
        # self.mBackEnd.debrief_cte(self.mModel_CTE)


    def addModelAggreagtes(self, table):
        lTrClass = self.mBestModel.mTransformation.__class__;
        lModelName = self.shorten("Model");
        TS1 = alias(select([ table.c[lModelName] , table.c[self.mRowNumberAlias] ]), "t_ModelComp");
        model_expr = TS1.c[ lModelName ]; # original signal
        rn_expr_1 = table.c[self.mRowNumberAlias];
        rn_expr_2 = TS1.c[self.mRowNumberAlias];
        cumulated_model = None;
        previous_model = None;
        if(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_None):
            pass;
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Accumulate):
            previous_model = select([func.sum(model_expr)]).where(rn_expr_1 == (rn_expr_2 + 1));
            pass
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Differencing):
            cumulated_model  = select([func.sum(model_expr)]).where(rn_expr_1 >= rn_expr_2);
            pass
        elif (lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_RelativeDifferencing):
            lMinExpr = self.as_float(tr.mTransformation.mMinValue);
            lDeltaExpr = self.as_float(tr.mTransformation.mDelta);
            # lNormalizedSignal = (sig_expr - lMinExpr) / lDeltaExpr;
            # lNormalizedLag1 = (previous_expr - lMinExpr) / lDeltaExpr;
            # relatve_diff = (lNormalizedSignal - lNormalizedLag1) / (lNormalizedLag1 + 1.0);
        else:
            assert(0);

        exprs = [];
        if(cumulated_model is not None):
            cumulated_model = cumulated_model.label("Cum_" + lModelName);
            exprs = [ cumulated_model ] + exprs;
            
        if(previous_model is not None):
            previous_model = previous_model.label("Lag1_" + lModelName);
            exprs = [ previous_model ] + exprs;
        
        return exprs

    def generateModelInvTransformationInputCode(self):
        exprs1 = self.addModelAggreagtes(self.mModel_CTE);
        self.mModelInvTransformationInputs_CTE = self.mBackEnd.generate_CTE(self.mModel_CTE.columns  + exprs1, "TS_CTE_ModelInvTransformationInput")
        # self.mBackEnd.debrief_cte(self.mModelInvTransformationInputs_CTE)

    def addTransformedModel(self, table):
        exprs = [];
        lTrClass = self.mBestModel.mTransformation.__class__;
        # lName = "Diff_"
        lModelName = self.shorten("Model");
        # TS1 = alias(select([ table.c[lModelName] , table.c[self.mRowNumberAlias] ]), "t_ModelAgg");
        model_expr = table.c[ lModelName ]; # original signal
        transformed_model = model_expr;
        if(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Accumulate):
            lDefault_expr = self.as_float(tr.mTransformation.mFirstValue);            
            previous_expr = func.coalesce(table.c["Lag1_" + lModelName] , lDefault_expr);
            transformed_model  = model_expr - previous_expr;
        elif(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_Differencing):
            lDefault_expr = self.as_float(tr.mTransformation.mFirstValue);            
            cum_expr = func.coalesce(table.c["Cum_" + lModelName] , lDefault_expr);
            transformed_model  = cum_expr;
        elif(lTrClass == pyaf.TS.Signal_Transformation.cSignalTransform_RelativeDifferencing):
            lDefault_expr = self.as_float(tr.mTransformation.mFirstValue)
            lDefault_expr = (lDefault_expr - tr.mTransformation.mMinValue) / tr.mTransformation.mDelta;
            lRelDiff = func.coalesce(table.c["RelDiff_" + self.mOriginalSignal] , lDefault_expr);
            transformed_model  = lRelDiff;
        else:
            assert(0);

        transformed_model = transformed_model.label("OriginalForecast");
        transformed_model_residue =  transformed_model - table.c[ self.mOriginalSignal ];
        transformed_model_residue = transformed_model_residue.label("OriginalForecast_Residue");
        exprs = exprs + [ transformed_model , transformed_model_residue];
        return exprs

    def generateInverseTransformationCode(self):
        model_exprs = self.addTransformedModel(self.mModelInvTransformationInputs_CTE) 
        self.mModelInvTransformation_CTE = self.mBackEnd.generate_CTE([self.mModelInvTransformationInputs_CTE]  + model_exprs, "TS_CTE_InvTransformedModel")
        # self.mBackEnd.debrief_cte(self.mModelInvTransformation_CTE)


    def generate_next_time(self, current, h):
        lTimeInfo = self.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            lNbSeconds = str(int(h * lTimeInfo.mTimeDelta.total_seconds()));
            lIntervalExpr = sqlalchemy.sql.expression.literal_column(lNbSeconds, sqlalchemy.types.String);
            lNextDateExpr = func.date_add(current , lIntervalExpr);
            return lNextDateExpr;        
        lIntervalExpr = self.as_float(h * lTimeInfo.mTimeDelta);
        lNextDateExpr = current + lIntervalExpr;
        return lNextDateExpr;
        

    def generateForecasts(self):
        base_table = self.mFinal_CTE;
        table = alias(select([base_table]) , "base_table_with_horizon_0");
        H = self.getHorizon();
        lCTEs = {};
        lCTEs [ 0 ] = base_table; 
        for h in range(1, H+1):
            # lRowTypeExpr = self.mBackEnd.getIntegerLiteral(h).label(self.mRowTypeAlias);
            table2 = alias(select([lCTEs[ h - 1 ]]) , "base_table_with_horizon_" + str(h-1) + "_1");
            lPreviousDate = select([ func.max(table2.c[self.mDateName]) ]).as_scalar();
            lNextDateExpr = self.generate_next_time(lPreviousDate , 1);
            lNextDateExpr = lNextDateExpr.label(self.mDateName);
            empty_line = [];
            # empty_line = empty_line + [lRowTypeExpr];
            empty_line = empty_line + [lNextDateExpr];
            for (key, col) in table.columns.items():
                if(key != self.mDateName):
                    empty_line = empty_line + [ func.cast(null(), col.type).label(key) ];
            lSignal = func.coalesce(table.c[self.mOriginalSignal], table.c["PredictedSignal"]);
            lSignal = lSignal.label(self.mOriginalSignal);
            lModifiedTableColumns = [table.c[self.mDateName] , lSignal];
            for (key, col) in table.columns.items()[2:]:
                lModifiedTableColumns = lModifiedTableColumns + [ col ]
            sel1 = select(lModifiedTableColumns);
            sel2 = select(empty_line);
            # print(sel1.columns);
            # print(sel2.columns);
            lUnion = sel1.union_all(sel2);
            lUnion = alias(lUnion, "FC_Union_" + str(h))
            # print("HORIZON_UNION" , h , lUnion.columns);
            # print("HORIZON_UNION_TYPES" , h , [col.type for col in lUnion.columns]);
            lCTEs[ h ] = self.mBackEnd.generate_CTE([lUnion], "TS_CTE_Forecast_" + str(h))
            table = alias( select([lCTEs[ h ]]) , "base_table_with_horizon" + str(h));
            
        self.mForecast_CTE = lCTEs[ H ]
        # self.mBackEnd.debrief_cte(self.mForecast_CTE)

