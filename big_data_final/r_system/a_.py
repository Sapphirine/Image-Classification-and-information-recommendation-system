from flask import Blueprint
main = Blueprint('main', __name__)
 
import json
from r_g import engine
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
from flask import Flask, request
 
@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_ratings(user_id, count):
    top_ratings = recommendation_engine.get_top_ratings(user_id,count)
    return json.dumps(top_ratings)
 
@main.route("/<int:user_id>/ratings/<int:article_id>", methods=["GET"])
def article_ratings(user_id, article_id):
    ratings = recommendation_engine.get_ratings_for_article_ids(user_id, [article_id])
    return json.dumps(ratings)
 
 
@main.route("/add_ratings")
def add_ratings():
    recommendation_engine.add_rating()
    return ""
 
def create_app(spark_context, dataset_path):
    global recommendation_engine 
    recommendation_engine = engine(spark_context, dataset_path)    
    app = Flask(__name__)
    app.register_blueprint(main)
    return app 
