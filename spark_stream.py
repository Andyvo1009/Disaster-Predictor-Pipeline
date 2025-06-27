from pyspark.sql import SparkSession
import os
# Create directories if they don't exist
os.makedirs("output/stream", exist_ok=True)
os.makedirs("output/checkpoint", exist_ok=True)

# Create Spark session WITHOUT the kafka package in the config
# We'll use spark-submit to provide that instead
spark = SparkSession.builder \
    .appName("KafkaToFile") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("DEBUG")

print("Created Spark session, connecting to Kafka...")

try:
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "test-tweets") \
        .option("startingOffsets", "earliest") \
        .load()
        
    print("Connected to Kafka, processing messages...")

    # Extract the message content
    messages = df.selectExpr("CAST(value AS STRING) as message")
    
    # Write to console first
    print("Writing to console...")
    console_query = messages.writeStream \
        .format("console") \
        .option("truncate", "false") \
        .start()
    
    # Let console run for a bit 
    console_query.awaitTermination(10)
    print("Console output completed, now writing to file...")
    
    # Then write to file
    file_query = messages.writeStream \
        .format("text") \
        .option("path", "output/stream") \
        .option("checkpointLocation", "output/checkpoint") \
        .outputMode("append") \
        .start()
        
    print("File output started, awaiting termination...")
    file_query.awaitTermination()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()