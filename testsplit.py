from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

spark = SparkSession.builder.appName("My_Session").getOrCreate()


goodreads_interactions = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
goodreads_interactions.createOrReplaceTempView('goodreads_interactions')


from imp import reload
import queries
reload(queries)

result_interaction =  queries.top_interaction(spark, 'hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv')

result_interaction.write.parquet('hdfs:/user/pg1910/pub/goodreads/goodreads_interactions_1p.parquet')
data_parquet = spark.read.parquet('hdfs:/user/pg1910/pub/goodreads/goodreads_interactions_1p.parquet')
# print(data_parquet.count())
# data_parquet.show()

train,val,test = data_parquet.randomSplit(weights=[0.6, 0.2, 0.2], seed=36)

result_sort = val.orderBy(val.user_id.asc())
result_sort = result_sort.withColumn("columnx",lit("ABC"))

w = Window().partitionBy('columnx').orderBy(lit('A'))
df = result_sort.withColumn("NumberofRows", row_number().over(w)).drop("columnx")
df.createOrReplaceTempView('df')
df_train = spark.sql('SELECT * FROM df WHERE NumberofRows%2=0')
df_train.createOrReplaceTempView('df_train')
df_val = spark.sql('SELECT * FROM df WHERE NumberofRows%2=1')
df_val.createOrReplaceTempView('df_val')
df_train = df_train.drop('NumberofRows')
val = df_val.drop('NumberofRows')

train = train.union(df_train)

result_test = test.orderBy(test.user_id.asc())
result_test = result_test.withColumn("columnx",lit("ABC"))

wt = Window().partitionBy('columnx').orderBy(lit('A'))
df2 = result_test.withColumn("NumberofRows", row_number().over(w)).drop("columnx")
df2.createOrReplaceTempView('df2')
dft_train = spark.sql('SELECT * FROM df2 WHERE NumberofRows%2=0')
dft_train.createOrReplaceTempView('dft_train')
dft_test = spark.sql('SELECT * FROM df2 WHERE NumberofRows%2=1')
dft_test.createOrReplaceTempView('dft_test')
dft_train = dft_train.drop('NumberofRows')
test = dft_test.drop('NumberofRows')

train = train.union(dft_train)

# print(train.count())
# print(test.count())
# print(val.count())

train.write.parquet('hdfs:/user/pg1910/pub/goodreads/training_sample_1p.parquet')
val.write.parquet('hdfs:/user/pg1910/pub/goodreads/validation_sample_1p.parquet')
test.write.parquet('hdfs:/user/pg1910/pub/goodreads/testing_sample_1p.parquet')

