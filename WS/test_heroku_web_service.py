# PYAF_URL="http://pyaf.herokuapp.com/model"
PYAF_URL="http://0.0.0.0:8081/model"

def test_heroku_pyaf(data):
    url = PYAF_URL
    header = {"Content-Type":"application/json"}

    import httplib2 
    http = httplib2.Http()
    response, send = http.request(url, "POST", headers=header, body=data)
    content = response.read()
    return content;


def test_heroku_pyaf_2(data):
    import json, urllib3

    http = urllib3.PoolManager()

    r = http.request('POST', PYAF_URL,
                     headers={'Content-Type': 'application/json'},
                     body=json.dumps(data))

    content = r.data
    return content;

CSV="https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"
data={"Name":"model1", "CSVFile":CSV, "DateFormat":"%Y-%m"}
cont = test_heroku_pyaf_2(data);
print(cont);

data2={"CSVFile":CSV, "DateFormat":"%Y-%m"}
cont1 = test_heroku_pyaf_2(data2);
print(cont1);

