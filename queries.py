#Query to downsample 1% of the the actual dataset
def top_interaction(spark, file_path):

    # TODO:
    file = spark.read.csv(file_path, header=True, schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
    file.createOrReplaceTempView('file')

    results = spark.sql('SELECT * FROM file WHERE rating != 0')
    results.createOrReplaceTempView('results')
    resultsintr = spark.sql('SELECT * FROM results WHERE user_id IN (SELECT user_id FROM results GROUP BY user_id HAVING COUNT(user_id) > 10)')
    resultsintr.createOrReplaceTempView('resultsintr')
    result1 = spark.sql('SELECT * FROM resultsintr WHERE user_id%100 = 0')
    result1.createOrReplaceTempView('result1')
    return result1


#Query to downsample 10% of the the actual dataset
def top_interaction10p(spark, file_path):

    # TODO:
    file = spark.read.csv(file_path, header=True, schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
    file.createOrReplaceTempView('file')

    results = spark.sql('SELECT * FROM file WHERE rating != 0')
    results.createOrReplaceTempView('results')
    print('Rating not 0:',results.count())
    resultsintr = spark.sql('SELECT * FROM results WHERE user_id IN (SELECT user_id FROM results GROUP BY user_id HAVING COUNT(user_id) > 10)')
    resultsintr.createOrReplaceTempView('resultsintr')
    result1 = spark.sql('SELECT * FROM resultsintr WHERE user_id%10 = 0')
    print('10% data:',result1.count())
    result1.createOrReplaceTempView('result1')
    return result1

