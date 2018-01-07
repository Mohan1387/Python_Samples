import pandas as pd
import urllib2 as url
import json
import sys
import datetime

for line in sys.stdin:
 line = line.strip()
 req = url.Request("http://api.bulkwhoisapi.com/whoisAPI.php?domain="+line+"&token=usemeforfree")
 response = url.urlopen(req)
 the_page = response.read()
 if(the_page != 'Invalid Input!'):
  res_dict = json.loads(the_page)
  if(res_dict['response_code'] != "failure"):
   creation_date = res_dict['formatted_data']['CreationDate']
   #print(creation_date[:10].encode('utf8'))
   sys.stdout.write(creation_date[:10].encode('utf8'))
  else:
   #print("9999-01-01")
   sys.stdout.write("9999-01-01")    
 else:
  #print("9999-01-01")
  sys.stdout.write("9999-01-01")
