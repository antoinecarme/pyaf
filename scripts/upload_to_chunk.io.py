import os
myhost = os.uname()[1]
print(myhost)
if("testing-docker" in myhost):
    print("on travis-ci")
    os.system("tar cvfz last_logs.tar.gz logs/")
    os.system("curl -s -T last_logs.tar.gz chunk.io")

