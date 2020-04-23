

# rebuild the wheel => dist/pyaf-1.2.2-py2.py3-none-any.whl
python3 setup.py bdist_wheel --universal

# upload to pypi
twine upload dist/*

