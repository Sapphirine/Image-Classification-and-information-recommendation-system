from sqlalchemy import *
import csv
from sqlalchemy.pool import NullPool
#from flask import Flask
#from flask import Flask, flash, redirect, render_template, request, session, abort,g

#id=1 4981
#cat 4980
#train 8804

#chair 13729
#airplane 29759
#sofa 29779
#'airplane','train','cat','chair','sofa','tvmonitor'])

DATABASEURI = "postgresql://localhost/big_data"
engine = create_engine(DATABASEURI)
try:
    conn = engine.connect()
except:
    print "uh oh, problem connecting to database"
    import traceback;
    traceback.print_exc()
    conn = None

id = 1
label = 'cat'
print "4981"
with open("/Users/jenan/Documents/big_data_final/dataset/Cats.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id


id = 4981
label = 'train'
with open("/Users/jenan/Documents/big_data_final/dataset/train.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id

id = 8805
label = 'chair'
print "4981"
with open("/Users/jenan/Documents/big_data_final/dataset/chairs.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id

id = 13730
label = 'airplane'
print "4981"
with open("/Users/jenan/Documents/big_data_final/dataset/plane.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id

id = 29760
label = 'sofa'
print "4981"
with open("/Users/jenan/Documents/big_data_final/dataset/sofa.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id

id = 29780
label = 'tvmonitor'
print "4981"
with open("/Users/jenan/Documents/big_data_final/dataset/television.csv", 'r') as csvin:
    reader = csv.reader(csvin)
    for item in reader:
        if reader.line_num == 1:
            continue;
        title = item[0]
        content = item[2]
        title = title.replace("'"," ")
        title = title.replace("%", " ")
        content = content.replace("'"," ")
        content = content.replace("%"," ")
        print content
        cursor = conn.execute("INSERT INTO wiki(id,label,title,content) VALUES("+str(id)+",'"+str(label)+"','"+str(title)+"','"+str(content)+"')")
        id = id+1
        print (id,cursor)
    cursor.close()
print id

