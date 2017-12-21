import time, sys, cherrypy, os
from paste.translogger import TransLogger
from a_ import create_app
from pyspark import SparkContext, SparkConf

def init_spark_context():# load spark context
    conf = SparkConf().setAppName("server")
    sc = SparkContext(conf=conf, pyFiles=['r_g.py', 'a_.py'])
    return sc
 
def run_server(app):
    app_logged = TransLogger(app)
    cherrypy.tree.graft(app_logged, '/')
    cherrypy.config.update({'server.socket_port': 3030,})
    cherrypy.engine.start()
    cherrypy.engine.block() 
 
if __name__ == "__main__":
    sc = init_spark_context()
    dataset_path = os.path.join('datasets', 'ml-latest')
    app = create_app(sc, dataset_path)
    run_server(app)

