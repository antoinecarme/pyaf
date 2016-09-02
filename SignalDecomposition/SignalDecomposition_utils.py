import sys, os

def createDirIfNeeded(dirname):
    try:
        os.mkdir(dirname);
    except:
        pass



class ForecastError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason

class InternalForecastError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason

