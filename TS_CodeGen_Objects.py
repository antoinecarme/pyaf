from sqlalchemy.engine import reflection

import sqlalchemy
from sqlalchemy import *
from sqlalchemy.sql import column
from sqlalchemy.pool import NullPool

from sqlalchemy.dialects import *



class cDecompositionCodeGenObject:

    def __init__(self):
        self.mAutoForecast = None;
        self.mMeta = MetaData();
        self.mDialect = sqlalchemy.dialects.sqlite.dialect();
        pass

    def generate_Sql(self, statement):
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

    def getFloatLiteral(self, iValue):
        return sqlalchemy.sql.expression.literal(iValue, sqlalchemy.types.FLOAT);


    def generateCode(self, iAutoForecast):
        self.mAutoForecast = iAutoForecast;
        self.generateCode_Internal();
        statement = alias(select([self.mModel_CTE]) , "SQLGenResult");
        return self.generate_Sql(statement);

    def generateCode_Internal(self):
         # M = T + C + AR
         # 0. the input is a table Table1 containing he date and the signal, normalized date and row number
         # 0.1 compute the transformed sgnal in a CTE, CTE_Transformation, contains Table1
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

         self.Shortened = {};
         self.Shortened[self.mTrend.mOutName] = "STrend";
         self.Shortened[self.mCycle.mOutName] = "SCycle";
         self.Shortened[self.mAR.mOutName] = "SAR";
         self.Shortened[self.mCycle.getCycleName()] = "SCycle";
         self.Shortened[self.mCycle.getCycleResidueName()] = "SCycleRes";
         self.Shortened[self.mModelName] = "SModel";
         
         table = self.createLogicalTable("TestTableForCodeGen");         
         self.generateTransformationCode(table); # => Transformation_CTE
         self.generateTrendInputCode(); # => Trend_Inputs_CTE
         self.generateTrendCode(); # => Trend_CTE
         self.generateCycleInputCode(); # => Cycle_Inputs_CTE
         self.generateCycleCode(); # => Cycle_CTE
         self.generateCycleResidueCode(); # => Cycle_Residue_CTE
         self.generateARInputCode(); # => AR_Inputs_CTE
         self.generateARCode(); # => AR_CTE
         self.generateModelCode(); # => Model_CTE

    def addTransformedSignal(self, table):
        exprs = []
        trasformed_signal = table.c[ self.mSignalName ];
        trasformed_signal = trasformed_signal.label("Signal");
        # trasformed_time = table.c[lTimeName];
        # year1 = func.extract('year' , trasformed_time);
        # year1 = func.cast(trasformed_time , Integer);
        # ratio = trasformed_time - year1
        # date2 = dt.datetime(year1, 7, 11, 12, 47, 28)
        # trasformed_time = func.dateadd(func.cast(trasformed_time, bindparam('tomorrow', timedelta(days=1), Interval()))
        # trasformed_signal = trasformed_signal.label("Time");
        normalized_time = None;
        lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
        if(lTimeInfo.isPhysicalTime()):
            # count the number of days here if date. number of seconds if datetime
            lAmplitude = func.cast(lTimeInfo.mTimeMax , DATETIME) - func.cast(lTimeInfo.mTimeMin , DATETIME)
            normalized_time = (func.cast(table.c[self.mDateName] , DATETIME) - func.cast(lTimeInfo.mTimeMin, DATETIME)) ;
        else:
            lAmplitude = lTimeInfo.mTimeMax - lTimeInfo.mTimeMin
            normalized_time = table.c[self.mDateName] - lTimeInfo.mTimeMin ;
        normalized_time = normalized_time / lAmplitude
        normalized_time = normalized_time.label("NTime")
        row_number_column = func.row_number().over(order_by=asc(table.c[self.mDateName])).label('RN') - 1
        row_number_column = row_number_column.label("RN")
        exprs = exprs + [ row_number_column , normalized_time, trasformed_signal ]; # , trasformed_time, year1, ratio, date2]
        return exprs

    def generateTransformationCode(self, table):
        # => Transformation_CTE
        signal_exprs = self.addTransformedSignal(table) 
        self.mTransformation_CTE = select([table] + signal_exprs).cte("CTE_Transformation")

    def julian_day(self, date_expr):
        expr_months = extract('year', date_expr) * 12 + extract('month', date_expr) 
        return expr_months;


    def addTrendInputs(self, table):
        lSecondsInADay = 3600.0 * 24;
        exprs = []
        normalized_time = table.c["NTime"];
        normalized_time_2 = normalized_time * normalized_time
        normalized_time_2 = normalized_time_2.label("NTime_2")
        normalized_time_3 = normalized_time_2 * normalized_time     
        normalized_time_3 = normalized_time_3.label("NTime_3")
        lag1 = self.getFloatLiteral(0.0);
        lag1 = lag1.label("Lag1")
        exprs = exprs + [ normalized_time_2, normalized_time_3, lag1]
        return exprs
    
    def generateTrendInputCode(self):
        # => Trend_Inputs_CTE
        trend_inputs = self.addTrendInputs(self.mTransformation_CTE) 
        self.mTrend_inputs_CTE = select([self.mTransformation_CTE] + trend_inputs).cte("TICTE")

    def generateTrendExpression(self, table):
        trend_expr = None;
        if(self.mTrend.mFormula == "ConstantTrend"):
            trend_expr = self.getFloatLiteral(self.mTrend.mMean);
            pass
        elif(self.mTrend.mFormula == "LinearTrend"):
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"] + self.mTrend.mTrendRidge.intercept_
            pass
        elif(self.mTrend.mFormula == "PolyTrend"):
            trend_expr = self.mTrend.mTrendRidge.coef_[0] * table.c["NTime"];
            trend_expr += self.mTrend.mTrendRidge.coef_[1] * table.c["NTime_2"];
            trend_expr += self.mTrend.mTrendRidge.coef_[2] * table.c["NTime_3"];
            trend_expr += self.mTrend.mTrendRidge.intercept_
            pass
        elif(self.mTrend.mFormula == "Lag1Trend"):
            trend_expr = table.c["Lag1"];
            pass
        return trend_expr;


    def addTrends(self, table):
        exprs = []
        trend_expr = self.generateTrendExpression(table);
        #print(type(trend_expr))
        
        trend_expr = trend_expr.label(self.Shortened[self.mTrend.mOutName]);
        exprs = exprs + [trend_expr]
        return exprs

    def generateTrendCode(self):
        # => Trend_CTE
        sel1 = alias(select([self.mTrend_inputs_CTE]), "TIN")
        # print(sel1.columns)
        trends = self.addTrends(sel1)
        self.mTrend_CTE = select([self.mTrend_inputs_CTE] + trends).cte("TCTE")



    def addCycleInputs(self, table):
        lTime =  self.mDateName;
        exprs = []
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
        sel1 = alias(select([self.mTrend_CTE]), "TCTE1")
        # print(sel1.columns)
        cycle_inputs = self.addCycleInputs(sel1)
        self.mCycle_input_CTE = select([self.mTrend_CTE] + cycle_inputs).cte("CICTE")


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
        exprs = []
        cycle_expr = self.generateCycleExpression(table);
        cycle_expr = cycle_expr.label(self.Shortened[self.mCycle.getCycleName()])
        exprs = exprs + [cycle_expr]
        return exprs
    
    def generateCycleCode(self):
        # => Cycle_CTE
        sel1 = alias(select([self.mCycle_input_CTE]), "CIN")
        # print(sel1.columns)
        cycles = self.addCycles(sel1)
        self.mCycle_CTE = select([self.mCycle_input_CTE] + cycles).cte("CYCTE")


    def addCycleResidues(self, table):
        exprs = []
        cycle_expr = table.c[self.Shortened[self.mCycle.getCycleName()]];
        trend_expr = table.c[self.Shortened[self.mTrend.mOutName]];
        cycle_residue_expr = trend_expr + cycle_expr - table.c[self.mSignalName]
        cycle_residue_expr = cycle_residue_expr.label(self.Shortened[self.mCycle.getCycleResidueName()])
        exprs = exprs + [cycle_residue_expr]
        return exprs


    def generateCycleResidueCode(self):
        # => Cycle_Residue_CTE
        sel1 = alias(select([self.mCycle_CTE]), "CYIN")
        # print(sel1.columns)
        cycle_resdiues = self.addCycleResidues(sel1)
        self.mCycle_residues_CTE = select([self.mCycle_CTE] + cycle_resdiues).cte("CYRESCTE")



    def createLags(self, table , H , col, index_col):
        TS = table
        TS1 = alias(table, "t");
        # TS2 = text(TS1);
        col_expr_1 = TS1.c[col];
        index_expr = TS.c[index_col]
        index_expr_1 = TS1.c[index_col]
        exprs = [table];
        for h in range(1 , H+1):
            expr1 = select([col_expr_1]).where(index_expr == (index_expr_1 + h));
            expr = expr1;
            expr = expr.label(col + "_Lag" + str(h));
            exprs = exprs + [expr];
        return exprs;

    def addARInputs(self, table):
        exprs = []
        if(self.mAR.mFormula != "NoAR"):
            residue_name = self.Shortened[self.mCycle.getCycleResidueName()];
            exprs = self.createLags(table, 
                                    len(self.mAR.mARLagNames), 
                                    residue_name,
                                    "RN");
        return exprs


    def generateARInputCode(self):
        # => AR_Inputs_CTE
        sel1 = alias(select([self.mCycle_residues_CTE]), "CYRESCTE1")
        # print(sel1.columns)
        ar_inputs = self.addARInputs(sel1)
        self.mAR_input_CTE = select([self.mCycle_residues_CTE] + ar_inputs).cte("ARICTE")

    def addARModel(self, table):
        exprs = table.columns
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
            ar_expr = ar_expr + self.mAR.mARRidge.intercept_;
        else:
            ar_expr = self.getFloatLiteral(0.0);
        ar_expr = ar_expr.label(self.Shortened[self.mAR.mOutName])
        exprs = exprs + [ar_expr]
        return exprs

    def generateARCode(self):
        # => AR_CTE
        sel1 = alias(select([self.mAR_input_CTE]), "ARI")
        # print(sel1.columns)
        ars = self.addARModel(sel1)
        self.mAR_CTE = select([self.mAR_input_CTE] + ars).cte("ARCTE")




    def add_TS_Model(self, table):
        exprs = table.columns
        sum_1 = table.c[self.Shortened[self.mTrend.mOutName]];
        sum_1 += table.c[self.Shortened[self.mCycle.mOutName]];
        sum_1 += table.c[self.Shortened[self.mAR.mOutName]];
        model_expr = sum_1;
        model_expr = model_expr.label(self.Shortened[self.mModelName])
        # model_residue = sum_1 - table.c[self.mSignalName]
        model_residue = model_expr - table.c[self.mSignalName]
        model_residue = model_residue.label(self.Shortened[self.mModelName] + "Residue")
        exprs = exprs + [model_expr , model_residue]
        return exprs

    def generateModelCode(self):
        # => Model_CTE
        sel1 = alias(select([self.mAR_CTE]), "AR")
        # print(sel1.columns)
        model_vars = self.add_TS_Model(sel1)
        self.mModel_CTE = select([self.mAR_CTE] + model_vars).cte("MODCTE")

