
import time
import subprocess
import os
import requests
import urllib2
dir = os.path.dirname(__file__)
starttime=time.time()
# Simple routine to run a query on a database and print the results:
rank = 20
def doQuery() :
    while True:
        conn = psycopg2.connect( host="localhost", dbname="big_data" )
        cur = conn.cursor()
        cur.execute( "SELECT * FROM account" )
        conn.commit()      
        #for i in cur :
            #an = subprocess.Popen("curl --data-binary @user_ratings" + i[0]+".file http://localhost:3030/" + i[0]+ "/ratings",shell=True)
            #with open("rating/user_ratings"+i[0]+".file","w") as fo:
                #fo.write("")
            #break
        #r = requests.post("http://localhost:3030/add_ratings")
        response = urllib2.urlopen('http://localhost:3030/add_ratings')
        for i in cur:
            print "User: " + str(i[0])
            url = "http://localhost:3030/"+ i[0] +"/ratings/top/40"
            r = requests.get(url)
            data = r.json()
            for j in range(len(data)):
                conn.cursor().execute("UPDATE recommend SET title='" +str(data[j][0])+"' Where id=" + i[0] +"AND rank=" + str(j+1))  
                conn.commit()
        conn.close()
        time.sleep(100000.0 - ((time.time() - starttime) % 100000.0))

def init(conn):
    cur = conn.cursor()
    for j in range(10):
        for i in range(40):
            cur.execute("INSERT INTO recommend(id,title,rank) VALUES ("+str(j+1)+",'',"+str(i+1)+")")
            conn.commit()
    conn.close()

import psycopg2
myConnection = psycopg2.connect( host="localhost", dbname="big_data" )
#init(myConnection)
doQuery()
myConnection.close()

   

