

sample_doc:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute docs/sample_code.ipynb

intro:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute docs/PyAF_Introduction.ipynb

exogenous:
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute docs/PyAF_Exogenous.ipynb

all: sample_doc intro exogenous

