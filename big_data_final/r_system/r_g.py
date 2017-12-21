import os
from pyspark.mllib.recommendation import ALS

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)


class engine:

    def __count_and_average_ratings(self):
        article_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        article_ID_with_avg_ratings_RDD = article_ID_with_ratings_RDD.map(get_counts_and_averages)
        self.articles_rating_counts_RDD = article_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    def __train_model(self):
        self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)

    def __predict_ratings(self, user_and_article_RDD):
        predicted_RDD = self.model.predictAll(user_and_article_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.articles_titles_RDD).join(self.articles_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
        
        return predicted_rating_title_and_count_RDD
    
    def add_rating(self):
        print "36"
        #self.sc = sc
        dataset_path = os.path.join('datasets', 'ml-latest')
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
        articles_file_path = os.path.join(dataset_path, 'articles.csv')
        articles_raw_RDD = self.sc.textFile(articles_file_path)
        articles_raw_data_header = articles_raw_RDD.take(1)[0]
        self.articles_RDD = articles_raw_RDD.filter(lambda line: line!=articles_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.articles_titles_RDD = self.articles_RDD.map(lambda x: (int(x[0]),x[1])).cache()
        self.__count_and_average_ratings()

        # Train the model
        self.rank = 8
        self.seed = 5L
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.__train_model() 

    def get_ratings_for_article_ids(self, user_id, article_ids):
        requested_articles_RDD = self.sc.parallelize(article_ids).map(lambda x: (user_id, x))
        ratings = self.__predict_ratings(requested_articles_RDD).collect()

        return ratings
    
    def get_top_ratings(self, user_id, articles_count):
        user_unrated_articles_RDD = self.ratings_RDD.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()
        ratings = self.__predict_ratings(user_unrated_articles_RDD).filter(lambda r: r[2]>=25).takeOrdered(articles_count, key=lambda x: -x[1])
        return ratings

    def __init__(self, sc, dataset_path):
        self.sc = sc
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
        articles_file_path = os.path.join(dataset_path, 'articles.csv')
        articles_raw_RDD = self.sc.textFile(articles_file_path)
        articles_raw_data_header = articles_raw_RDD.take(1)[0]
        self.articles_RDD = articles_raw_RDD.filter(lambda line: line!=articles_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.articles_titles_RDD = self.articles_RDD.map(lambda x: (int(x[0]),x[1])).cache()
        self.__count_and_average_ratings()

        # Train the model
        self.rank = 8
        self.seed = 5L
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.__train_model() 
