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

    def generateCode(self, iAutoForecast):
        self.mAutoForecast = iAutoForecast;
        self.generateCode_Internal();
        statement = select([self.mModel_CTE]);
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

         self.Shortify[self.mTrend.mOutName] = "STrend";
         self.Shortify[self.mCycle.mOutName] = "SCycle";
         self.Shortify[self.mAR.mOutName] = "SAR";

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
        exprs = exprs + [ trasformed_signal ]; # , trasformed_time, year1, ratio, date2]
        return exprs

    def generateTransformationCode(self, table):
        # => Transformation_CTE
        signal_exprs = self.addTransformedSignal(table) 
        self.mTransformation_CTE = select([table] + signal_exprs).cte("CTE_Transformation")

    def julian_day(date_expr):
        expr_months = extract('year', date_expr) * 12 + extract('month', date_expr) 
        return expr_months;

    def julian_day(self, date_expr):
        expr_months = extract('year', date_expr) * 12 + extract('month', date_expr) 
        return expr_months;


    def addTrendInputs(self, table , time_col):
        exprs = []
        lTimeInfo = self.mAutoForecast.mSignalDecomposition.mBestTransformation.mTimeInfo
        lAmplitude = lTimeInfo.mTimeMax - lTimeInfo.mTimeMin
        lAmplitude = lAmplitude
        row_number_column = func.row_number().over(order_by=asc(table.c[time_col])).label('RN') - 1
        row_number_column = row_number_column.label("RN")
        #    normalized_time = func.datediff(text('month'), table.c[time_col] , lTimeInfo.mTimeMin) / func.datediff(text('month'), lTimeInfo.mTimeMax , lTimeInfo.mTimeMin)
        normalized_time = (self.julian_day(table.c[time_col]) - self.julian_day(lTimeInfo.mTimeMin)) ;
        normalized_time = normalized_time / lAmplitude
        normalized_time = normalized_time.label("NTime")
        normalized_time_2 = normalized_time * normalized_time
        normalized_time_2 = normalized_time_2.label("NTime_2")
        normalized_time_3 = normalized_time_2 * normalized_time     
        normalized_time_3 = normalized_time_3.label("NTime_3")
        exprs = exprs + [row_number_column , normalized_time, normalized_time_2, normalized_time_3]
        return exprs
    
    def generateTrendInputCode(self):
        # => Trend_Inputs_CTE
        trend_inputs = self.addTrendInputs(self.mTransformation_CTE, self.mDateName) 
        self.mTrend_inputs_CTE = select([self.mTransformation_CTE] + trend_inputs).cte("CTE_Trend_Inputs")


    def addTrends(self, table):
        exprs = []
        trend_expr = table.c["normalized_time"]
        #print(type(trend_expr))
        trend_expr = self.mTrend.mTrendRidge.coef_[0] * trend_expr + self.mTrend.mTrendRidge.intercept_
        trend_expr = trend_expr.label(self.Shortify[self.mTrend.mOutName]);
        exprs = exprs + [trend_expr]
        return exprs

    def generateTrendCode(self):
        # => Trend_CTE
        sel1 = alias(select([self.mTrend_inputs_CTE]), "TrendInputAlias")
        # print(sel1.columns)
        trends = self.addTrends(sel1)
        self.mTrend_CTE = select([self.mTrend_inputs_CTE] + trends).cte("trend_CTE")



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
        sel1 = alias(select([self.mTrend_CTE]), "Trend_CTE1")
        # print(sel1.columns)
        cycle_inputs = self.addCycleInputs(sel1)
        self.mCycle_input_CTE = select([self.mTrend_CTE] + cycle_inputs).cte("cycle_input_CTE")


    def addCycles(self, table):
        exprs = []
        cycle_expr = table.c["RN"] * 0.0;
        cycle_expr = cycle_expr.label(self.mCycle.getCycleName())
        exprs = exprs + [cycle_expr]
        return exprs
    
    def generateCycleCode(self):
        # => Cycle_CTE
        sel1 = alias(select([self.mCycle_input_CTE]), "CycleInputAlias")
        # print(sel1.columns)
        cycles = self.addCycles(sel1)
        self.mCycle_CTE = select([self.mCycle_input_CTE] + cycles).cte("cycle_CTE")


    def addCycleResidues(self, table):
        exprs = []
        cycle_expr = table.c[self.mCycle.getCycleName()];
        trend_expr = table.c[self.Shortify[self.mTrend.mOutName]];
        cycle_residue_expr = trend_expr + cycle_expr - table.c[self.mSignalName]
        cycle_residue_expr = cycle_residue_expr.label(self.mCycle.getCycleResidueName())
        exprs = exprs + [cycle_residue_expr]
        return exprs


    def generateCycleResidueCode(self):
        # => Cycle_Residue_CTE
        sel1 = alias(select([self.mCycle_CTE]), "CycleAlias")
        # print(sel1.columns)
        cycle_resdiues = self.addCycleResidues(sel1)
        self.mCycle_residues_CTE = select([self.mCycle_CTE] + cycle_resdiues).cte("cycle_residues_CTE")



    def createLags(self, table , H , col, index_col):
        TS = table
        TS1 = table.alias("t");
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
        residue_name = self.mCycle.getCycleResidueName();
        exprs = self.createLags(table, 
                           len(self.mAR.mARLagNames), 
                           residue_name,
                           "RN");
        exprs = exprs
        return exprs


    def generateARInputCode(self):
        # => AR_Inputs_CTE
        sel1 = alias(select([self.mCycle_residues_CTE]), "AR_CTE1")
        # print(sel1.columns)
        ar_inputs = self.addARInputs(sel1)
        self.mAR_input_CTE = select([self.mCycle_residues_CTE] + ar_inputs).cte("ar_input_CTE")

    def addARModel(self, table):
        exprs = table.columns
        ar_expr = None;
        i = 0 ;
        for feat in self.mAR.mARLagNames:
            if(ar_expr is None):
                ar_expr = self.mAR.mARRidge.coef_[i] * table.c[feat];
            else:
                ar_expr = ar_expr + self.mAR.mARRidge.coef_[i] * table.c[feat];
            i = i + 1;
        ar_expr = ar_expr + self.mAR.mARRidge.intercept_;
        ar_expr = ar_expr.label(self.Shortify[self.mAR.mOutName])
        exprs = exprs + [ar_expr]
        return exprs

    def generateARCode(self):
        # => AR_CTE
        sel1 = alias(select([self.mAR_input_CTE]), "ARInputAlias")
        # print(sel1.columns)
        ars = self.addARModel(sel1)
        self.mAR_CTE = select([self.mAR_input_CTE] + ars).cte("ar_CTE")




    def add_TS_Model(self, table):
        exprs = table.columns
        model_expr = table.c[self.Shortify[self.mTrend.mOutName]] + table.c[self.Shortify[self.mCycle.mOutName]] + table.c[self.Shortify[self.mAR.mOutName]];
        model_expr = model_expr.label(self.mModelName)
        model_residue = model_expr - table.c[self.mSignalName]
        model_residue = model_residue.label(self.mModelName + "Residue")
        exprs = exprs + [model_expr , model_residue]
        return exprs

    def generateModelCode(self):
        # => Model_CTE
        sel1 = alias(select([self.mAR_CTE]), "AR_Alias")
        # print(sel1.columns)
        model_vars = self.add_TS_Model(sel1)
        self.mModel_CTE = select([self.mAR_CTE] + model_vars).cte("model_CTE")

