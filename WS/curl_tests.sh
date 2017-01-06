PYAF_URL="http://pyaf.herokuapp.com/model"
# PYAF_URL="http://0.0.0.0:8081/model"
CONTENT_TYPE="Content-Type: application/json"

# create a model
CSV="https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/ozone-la.csv"
DATA="{\"Name\":\"model1\", \"CSVFile\":\""$CSV"\", \"DateFormat\":\"%Y-%m\"}"
echo $DATA
curl --header "$CONTENT_TYPE" -X POST --data "$DATA" "$PYAF_URL"
# the same dataset on pastebin
CSV1="http://pastebin.com/raw/QsGam3uS"
DATA1="{\"Name\":\"model1_paste_bin\", \"CSVFile\":\""$CSV1"\", \"DateFormat\":\"%Y-%m\"}"
echo $DATA1
curl --header "$CONTENT_TYPE" -X POST --data "$DATA1" "$PYAF_URL"

DATA2="{\"CSVFile\":\""$CSV"\", \"DateFormat\":\"%Y-%m\"}"
echo $DATA2
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
curl --header "$CONTENT_TYPE" -X POST --data "$DATA2" "$PYAF_URL"
