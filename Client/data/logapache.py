# s = '83.149.9.216 - - [17/May/2015:10:05:03 +0000] "GET /presentations/logstash-monitorama-2013/images/kibana-search.png HTTP/1.1" 200 203023 "http://semicomplete.com/presentations/logstash-monitorama-2013/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.77 Safari/537.36"'
# b = [item.strip() for item in s.split(' - - ')]
# a=b[1]
# r = [item.strip() for item in a.split('"')]
# f = [item for item in r if item != ""]
# f.insert(0,b[0])
# f[3:4] = f[3].split(' ')
# for i in f:
#     print(i,'\t')

from datetime import datetime
unix_time = 1631026800
dt = datetime.utcfromtimestamp(unix_time)
print(dt)

# Lấy thời gian hiện tại và chuyển sang dạng RFC 3339
current_time = datetime.utcnow().isoformat() + "Z"
print(current_time)