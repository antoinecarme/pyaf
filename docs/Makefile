

sample_doc:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --stdout --execute docs/sample_code.ipynb >  outputs/sample_code.ipynb

intro:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --stdout --execute docs/PyAF_Introduction.ipynb > outputs/PyAF_Introduction.ipynb

exogenous:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --stdout --execute docs/PyAF_Exogenous.ipynb > outputs/PyAF_Exogenous.ipynb

all: sample_doc intro exogenous

