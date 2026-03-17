from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode

# Create Spark session
spark = SparkSession.builder \
    .appName("Video Recommendation") \
    .master("local[*]") \
    .config("spark.driver.host", "0.0.0.0") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .getOrCreate()
# Load data
ratings = spark.read.csv("data/ml-latest-small/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("data/ml-latest-small/movies.csv", header=True, inferSchema=True)

# Split data
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

# Train ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop"
)

model = als.fit(train)

# Evaluate
predictions = model.transform(test)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE =", rmse)

# Print RMSE

output_path = "./output/rmse.txt"

with open(output_path, "w") as f:
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")

print(f"RMSE saved to {output_path}")

# Generate Top 10 recommendations
user_recs = model.recommendForAllUsers(10)

recs = user_recs \
    .withColumn("rec", explode("recommendations")) \
    .select("userId", "rec.movieId", "rec.rating")

final = recs.join(movies, "movieId")

final.show(20, truncate=False)

#spark.stop()
input("Press Enter to exit...")
