from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, expr
from pyspark.mllib.evaluation import RankingMetrics

spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()


df_training = spark.read.parquet('hdfs:/user/tb2517/pub/goodreads/training_sample_10p.parquet')
df_validation = spark.read.parquet('hdfs:/user/tb2517/pub/goodreads/validation_sample_10p.parquet')
df_test = spark.read.parquet('hdfs:/user/tb2517/pub/goodreads/testing_sample_10p.parquet')



als=ALS(userCol="user_id",itemCol="book_id",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)

param_grid = ParamGridBuilder().addGrid(als.rank, [15,25,35]).addGrid(als.maxIter, [5,8,10]).addGrid(als.regParam, [0.08,0.09,0.10]).build()
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")

cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
model=cv.fit(df_training)

best_model = model.bestModel

print("Tuned Hyperparameters:-------------")
print("Rank: ", best_model._java_obj.parent().getRank())
print("MaxIter: ", best_model._java_obj.parent().getMaxIter())
print("RegParam: ", best_model._java_obj.parent().getRegParam())

print("Recommendations: ------------------------------")
user_recs = best_model.recommendForAllUsers(500)
print(user_recs.count())


prediction_val = best_model.transform(df_validation)
print(" Predictions for validation dataset: ------------------------------")
prediction_val.show()


prediction_test = best_model.transform(df_test)
print(" Predictions for test dataset: ------------------------------")
prediction_test.show()


actual_val = df_validation.groupBy("user_id").agg(expr("collect_set(book_id) as books"))
pred_val = user_recs.select('user_id','recommendations.book_id')
output_val =pred_val.join(actual_val,['user_id']).select('book_id','books')
metrics_val = RankingMetrics(output_val.rdd)
result_val = metrics_val.meanAveragePrecision
result_val2 = metrics_val.precisionAt(20)

print("Mean average precision for validation dataset: " + str(result_val))
print("Precision @ 20 for validation dataset: " + str(result_val2))
rmse_val = evaluator.evaluate(prediction_val)
print("RMSE for validation dataset=" + str(rmse_val))


actual_test = df_test.groupBy("user_id").agg(expr("collect_set(book_id) as books"))
pred_test = user_recs.select('user_id','recommendations.book_id')
output_test =pred_test.join(actual_test,['user_id']).select('book_id','books')
metrics_test = RankingMetrics(output_test.rdd)
result_test = metrics_test.meanAveragePrecision
result_test2 = metrics_test.precisionAt(20)

print("Mean average precision for test dataset: " + str(result_test))
print("Precision @ 20 for test dataset: " + str(result_test2))
rmse_test = evaluator.evaluate(prediction_test)
print("RMSE for test dataset=" + str(rmse_test))
