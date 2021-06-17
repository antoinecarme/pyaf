import os
circleci = os.environ.get("CIRCLECI", None)
if(circleci == "true"):
    print("on circleci")
    os.system("tar cvfz last_logs.tar.gz logs/")
    os.system("curl -s -T last_logs.tar.gz curldu.mp")

