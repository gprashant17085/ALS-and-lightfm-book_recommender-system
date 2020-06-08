from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, expr
from pyspark.mllib.evaluation import RankingMetrics
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import numpy as np
import time
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
import time
from lightfm.evaluation import auc_score
import pandas as pd
import pickle
import re
import seaborn as sns

spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()

df_training = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/training_sample_1p.csv', schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
df_training.createOrReplaceTempView('df_training')

df_validation = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/validation_sample_1p.csv', schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
df_validation.createOrReplaceTempView('df_validation')

df_test = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/testing_sample_1p.csv', schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
df_test.createOrReplaceTempView('df_test')
#
# df_training_parquet = spark.read.parquet('hdfs:/user/pg1910/pub/goodreads/training_sample_1p.parquet')
# df_validation_parquet  = spark.read.parquet('hdfs:/user/pg1910/pub/goodreads/validation_sample_1p.parquet')
# df_test_parquet  = spark.read.parquet('hdfs:/user/pg1910/pub/goodreads/testing_sample_1p.parquet')
#
# df_training_parquet.write.csv('hdfs:/user/pg1910/pub/goodreads/training_sample_1p.csv')
# df_validation_parquet.write.csv('hdfs:/user/pg1910/pub/goodreads/validation_sample_1p.csv')
# df_test_parquet.write.csv('hdfs:/user/pg1910/pub/goodreads/testing_sample_1p.csv')

# df_training = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/training_sample_1p.csv')
# df_validation = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/validation_sample_1p.csv')
# df_test = spark.read.csv('hdfs:/user/pg1910/pub/goodreads/testing_sample_1p.csv')

# df_training = df_training.select(df_training['user_id'],df_training['book_id'],df_training['rating'])
# df_training.createOrReplaceTempView('df_training')
#
# df_validation = df_validation.select(df_validation['user_id'],df_validation['book_id'],df_validation['rating'])
# df_validation.createOrReplaceTempView('df_validation')
#
# df_test = df_test.select(df_test['user_id'],df_test['book_id'],df_test['rating'])
# df_test.createOrReplaceTempView('df_test')


df_training = df_training.select("*").toPandas()
df_validation = df_validation.select("*").toPandas()
df_test = df_test.select("*").toPandas()

def dataformatting(some_df):
    split_cut = np.int(np.round(some_df.shape[0]))
    res_df = some_df.iloc[0:split_cut]


    id_cols = ['user_id', 'book_id']
    trans_cat_res = dict()

    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder()
        trans_cat_res[k] = cate_enc.fit_transform(res_df[k].values)


    cate_enc = preprocessing.LabelEncoder()
    ratings = dict()
    ratings['res'] = cate_enc.fit_transform(res_df.rating)

    n_users = len(np.unique(trans_cat_res['user_id']))
    n_books = len(np.unique(trans_cat_res['book_id']))
    res = coo_matrix((ratings['res'], (trans_cat_res['user_id'], trans_cat_res['book_id'])), shape=(n_users, n_books))

    return res, res_df



train, raw_train_df = dataformatting(df_training)
test, raw_test_df = dataformatting(df_test)
validation, raw_validation_df = dataformatting(df_validation)

start_time = time.time()
model=LightFM(no_components=110,learning_rate=0.027,loss='warp')
model.fit(train, epochs=12,num_threads=4)

train_auc = auc_score(model, train).mean()
print("--- Run time:  {} mins ---".format((time.time() - start_time)/60))
print("Train AUC Score: {}".format(train_auc))
test_auc = auc_score(model, test).mean()
validation_auc = auc_score(model, validation).mean()
print("Test AUC  Score: {}".format(test_auc))
print("Validation AUC  Score: {}".format(validation_auc))

