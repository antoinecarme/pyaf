PYAF_URL="http://pyaf.herokuapp.com/model"
#PYAF_URL="http://0.0.0.0:8081/model"

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
    content = content.decode('utf-8').replace("\\n" , "\n");
    r.close()
    return content;

CSV="https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"
data={"Name":"model1", "CSVFile":CSV, "DateFormat":"%Y-%m", "Present": "1968-08", "Horizon":21}
cont = test_heroku_pyaf_2(data);
print(cont);


