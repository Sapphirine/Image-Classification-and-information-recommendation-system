#!/usr/bin/env python2.7

"""
To run locally:

    python server.py

Go to http://localhost:8111 in your browser.

A debugger such as "pdb" may be helpful for debugging.
Read about it online.
"""
import atexit
from apscheduler.scheduler import Scheduler
import os
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask,url_for
from flask import Flask, flash, redirect, render_template, request, session, abort,g
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user 
from werkzeug import secure_filename
import random
import time
import logging
import argparse
import glob
import cv2
import pickle
from sklearn.decomposition import PCA
from statistics import mode
from VLAD import *
from Descriptors import *
from scipy.stats import mode
logging.basicConfig()

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
QUERY_FOLDER = os.path.join('static', 'query')
app.config['QUERY_FOLDER'] = QUERY_FOLDER
cron = Scheduler(daemon=True)
UPLOAD_FOLDER = 'static/query'
ALLOWED_EXTENSIONS = set(['doc', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
classMap = list(['classValue','airplane','train','cat','chair','sofa','tvmonitor'])

# Explicitly kick off the background thread
cron.start()

DATABASEURI = "postgresql://localhost/big_data"
engine = create_engine(DATABASEURI)
@app.before_request
def before_request():
  """
  This function is run at the beginning of every web request 
  (every time you enter an address in the web browser).
  We use it to setup a database connection that can be used throughout the request.

  The variable g is globally accessible.
  """
  try:
    g.conn = engine.connect()
  except:
    print "uh oh, problem connecting to database"
    import traceback; traceback.print_exc()
    g.conn = None

@app.teardown_request
def teardown_request(exception):
  """
  At the end of the web request, this makes sure to close the database connection.
  If you don't, the database could run out of memory!
  """
  try:
    g.conn.close()
  except Exception as e:
    pass



@app.route('/')
def index(): 
  if not session.get('logged_in'):
  	return render_template('signIn.html')
  else:
  	return main()


@app.route('/login', methods=['POST'])
def do_login():
  id = request.form['id']
  password = request.form['password']
  cursor = g.conn.execute("SELECT count(*) FROM account Where username='"+id+"' And password='"+password+"'")
  count =  cursor.scalar()
  if count != 0:
  	cursor = g.conn.execute("SELECT * FROM account Where username='"+id+"' And password='"+password+"'")
  	session['logged_in'] = True
  	session['id'] = id
  else:
  	flash('wrong password!')
  return index()



@app.route('/main')
def main():
  if not session.get('category'):
    return render_template('index.html')
  else:
    label = session['category']
    cursor = g.conn.execute("SELECT * FROM wiki where label='"+label+"' ORDER BY RANDOM() LIMIT 10")
    #newcursor = g.conn.execute("SELECT * FROM recommend R wiki w where id=" + str(session['id']))
    #recommendlist =[]
    #for k in newcursor:
    #  title =k[1]
    # print title
    #  numberOfdata = g.conn.execute("SELECT count(*) FROM wiki where label ='"+label+"' AND title='" + str(title) +"'" )
    #  lenOfdata = numberOfdata.scalar()
    # print lenOfdata
    #  nC = g.conn.execute("SELECT * FROM wiki where label ='"+label+"' AND title='" + str(title) +"' LIMIT 1" )
    #  if lenOfdata != 0:
    #    for h in nC:
    #      d = {'id': h[0], 'title': h[2]}
    #      recommendlist.append(dict(d))
    #print recommendlist
    full_filename = os.path.join(app.config['QUERY_FOLDER'],  session['f_name'])
    newcursor = g.conn.execute("SELECT R.title,Min(W.id) FROM recommend R, wiki W where R.id="+str(session['id'])+" AND R.title = W.title And label ='"+label+"' GROUP BY R.title,R.rank ORDER BY R.rank")
    return render_template('index.html',randomList=cursor, recommendList = newcursor,label=label,path=full_filename)

@app.route('/id/<int:id>')
def rating(id):
	cursor = g.conn.execute("SELECT * FROM wiki where id='"+str(id)+"'")
	session['article_id'] = id
	return render_template('rating.html',id=id, content=cursor)

@app.route('/addRating', methods=['POST'])
def addRating():
  id = session['id']
  article_id = session['article_id']
  star = request.form['rating']
  #with open("rating/user_ratings"+id+".file","a") as fo:
  #		fo.write(str(article_id)+","+str(star)+"\n")
  fd = open('r_system/datasets/ml-latest/ratings.csv', 'a')
  fd.write(str(id) + "," + str(article_id) + "," + str(float(star)) +"," "\n")
  fd.close()
  return index()

	
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename("file.filename")
            f_name= str(random.randint(1,1010000)) + ".jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
            while not os.path.exists("static/query/"+f_name):
            	time.sleep(30)
            imageClassfication(f_name)
    return index()


def imageClassfication(f_name):
  queryPath = 'static/query/' + f_name
  session['f_name'] = f_name
  print queryPath
  descriptorName = 'SURF'
  k = 1
  treeIndex = 'ballTreeIndexes/ballTreeIndexes.pickle'
  pathVD = 'visualDictionary/visualDictionary.pickle'
  #load the index
  with open(treeIndex, 'rb') as f:
    indexStructure=pickle.load(f)

	#load the visual dictionary
  with open(pathVD, 'rb') as f:
    visualDictionary=pickle.load(f)     

  imageID=indexStructure[0]
  tree = indexStructure[1]
  pathImageData = indexStructure[2]
  imageClasses = indexStructure[3]
	#computing descriptors
  dist,ind = query(queryPath, k, descriptorName, visualDictionary, tree)

  ind=list(itertools.chain.from_iterable(ind))

  print queryPath 
  results = list()
  for i in ind:
    results = np.hstack((results,imageClasses[i]))
    print imageID[i]
  print 'the query image class id is:'
  print mode(results).mode[0]
  classMap = list(['classValue','airplane','train','cat','chair','sofa','tvmonitor'])
  print classMap[int(mode(results).mode[0])]
  session['category'] = classMap[int(mode(results).mode[0])]







if __name__ == "__main__":
  import click
  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:

        python server.py

    Show the help text using:

        python server.py --help

    """

    HOST, PORT = host, port
    print "running on %s:%d" % (HOST, PORT)
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
  run()




