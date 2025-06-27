"""
Spark Streaming Script for Disaster Prediction using Custom Trained BERT Model

This script reads text data from test.csv and uses your trained BERT model 
(located in bert_model/ folder) to predict whether the text is about a disaster.

The model uses PEFT (LoRA) adapters trained on your custom dataset.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, monotonically_increasing_id
from pyspark.sql.types import StringType, IntegerType
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
import pyspark
# Create directories if they don't exist
os.makedirs("output/stream", exist_ok=True)
os.makedirs("output/checkpoint", exist_ok=True)
os.makedirs("output/predictions", exist_ok=True)
class DisasterPredictor:
    """Class to handle BERT model loading and prediction"""
    
    def __init__(self, model_path="bert_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load your trained tokenizer from the model directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model first
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-uncased",
            num_labels=2  # Binary classification for disaster/non-disaster
        )
        
        # Load your trained PEFT adapter
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Your trained BERT model loaded successfully from {model_path}!")
        print(f"Model type: {type(self.model)}")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        # Print adapter info
        if hasattr(self.model, 'peft_config'):
            print(f"PEFT adapter type: {self.model.peft_config}")
        
        # Test with a sample prediction to ensure everything works
        try:
            test_text = "There was an earthquake in the city"
            test_pred, test_conf = self.predict_disaster(test_text)
            print(f"Test prediction - Text: '{test_text}' -> Disaster: {test_pred}, Confidence: {test_conf:.4f}")
        except Exception as e:
            print(f"Warning: Test prediction failed: {e}")
    
    def predict_disaster(self, text):
        """Predict if text is about a disaster using your trained model"""
        try:
            if not text or text.strip() == "":
                return 0, 0.0
                
            # Tokenize input using your trained tokenizer
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction from your trained model
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Return prediction (1 = disaster, 0 = not disaster)
            return int(predicted_class), float(confidence)
            
        except Exception as e:
            print(f"Error predicting for text: '{text[:50] if text else 'None'}...' Error: {str(e)}")
            return 0, 0.0  # Default to non-disaster with 0 confidence

# Initialize the predictor (singleton pattern)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = DisasterPredictor()
    return predictor

def predict_disaster_udf(text):
    """UDF wrapper for disaster prediction"""
    if text is None or text.strip() == "":
        return "0,0.0"  # Return as string to be parsed later
    
    pred = get_predictor()
    prediction, confidence = pred.predict_disaster(text)
    return f"{prediction},{confidence:.4f}"

# Register UDF
predict_udf = udf(predict_disaster_udf, StringType())

def create_spark_session():
    """Create Spark session with appropriate configuration"""
    return SparkSession.builder \
        .appName("CSVBERTDisasterPrediction") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def process_csv_stream():
    """Process CSV file in streaming fashion"""
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print("Created Spark session, reading CSV file...")
    
    try:
        # Read CSV file as a static DataFrame first
        csv_path = "data/test.csv"
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
        
        print(f"Loaded CSV with {df.count()} rows")
        
        # Select only the text column and filter out null values
        text_df = df.select("text").filter(col("text").isNotNull())
        
        print("Processing texts with BERT model...")
        
        # Add predictions using UDF
        predictions_df = text_df.withColumn("prediction_result", predict_udf(col("text")))
        
        # Split the prediction result into separate columns
        from pyspark.sql.functions import split
        final_df = predictions_df \
            .withColumn("prediction", split(col("prediction_result"), ",")[0].cast(IntegerType())) \
            .withColumn("confidence", split(col("prediction_result"), ",")[1].cast("double")) \
            .drop("prediction_result") \
            .withColumn("id", monotonically_increasing_id())
        
        # Add disaster label
        from pyspark.sql.functions import when
        final_df = final_df.withColumn(
            "disaster_label",
            when(col("prediction") == 1, "DISASTER").otherwise("NOT_DISASTER")
        )
        
        print("Predictions completed. Writing results...")
        
        # Write results to console
        print("\n=== SAMPLE PREDICTIONS ===")
        final_df.select("text", "disaster_label", "confidence").show(20, truncate=False)
        
        # Write to file
        final_df.coalesce(1).write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv("output/predictions")
        
        print("Results saved to output/predictions/")
        
        # Show summary statistics
        print("\n=== PREDICTION SUMMARY ===")
        summary = final_df.groupBy("disaster_label").count().orderBy("count", ascending=False)
        summary.show()
        
        # Show high confidence disaster predictions
        print("\n=== HIGH CONFIDENCE DISASTER PREDICTIONS ===")
        high_conf_disasters = final_df.filter(
            (col("prediction") == 1) & (col("confidence") > 0.8)
        ).select("text", "confidence").orderBy("confidence", ascending=False)
        
        high_conf_disasters.show(10, truncate=False)
        
    except Exception as e:
        print(f"Error processing stream: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()

def process_streaming_csv():
    """Alternative approach: Process CSV in batches to simulate streaming"""
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print("Starting batch processing simulation...")
    
    try:
        # Read CSV file as a static DataFrame first
        csv_path = "data/test.csv"
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
        
        print(f"Loaded CSV with {df.count()} rows")
        
        # Process in batches to simulate streaming
        batch_size = 5  # Process 5 rows at a time
        total_rows = df.count()
        
        print(f"Processing {total_rows} rows in batches of {batch_size}...")
        
        for i in range(0, total_rows, batch_size):
            batch_num = (i // batch_size) + 1
            print(f"\n--- Processing Batch {batch_num} (rows {i+1} to {min(i+batch_size, total_rows)}) ---")
            
            # Get batch of data
            batch_df = df.limit(i + batch_size).subtract(df.limit(i)) if i > 0 else df.limit(batch_size)
            
            # Select only the text column and filter out null values
            text_df = batch_df.select("text").filter(col("text").isNotNull())
            
            if text_df.count() == 0:
                print("No valid text data in this batch, skipping...")
                continue
            
            # Add predictions using UDF
            predictions_df = text_df.withColumn("prediction_result", predict_udf(col("text")))
            from pyspark.sql.functions import when, lit
            # Split the prediction result into separate columns
            from pyspark.sql.functions import split
            final_df = predictions_df \
                .withColumn("prediction", split(col("prediction_result"), ",")[0].cast(IntegerType())) \
                .withColumn("confidence", split(col("prediction_result"), ",")[1].cast("double")) \
                .drop("prediction_result") \
                .withColumn("batch_id", lit(batch_num))
            
            # Add disaster label
            
            final_df = final_df.withColumn(
                "disaster_label",
                when(col("prediction") == 1, "DISASTER").otherwise("NOT_DISASTER")
            )
            
            # Show batch results
            print(f"Batch {batch_num} predictions:")
            final_df.select("text", "disaster_label", "confidence").show(3, truncate=False)
            
            # Save batch results
            final_df.coalesce(1).write \
                .mode("append") \
                .option("header", "true") \
                .csv(f"output/stream/batch_{batch_num}")
            
            print(f"Batch {batch_num} saved to output/stream/batch_{batch_num}/")
            
            # Simulate streaming delay
            import time
            time.sleep(2)  # Wait 2 seconds between batches
        
        print(f"\nStreaming simulation completed! Processed {total_rows} rows in {batch_num} batches")
        
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error in streaming: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    print("BERT Disaster Prediction with Spark")
    print("====================================")
    print("Using YOUR trained BERT model from bert_model/ folder")
    print("Model: BERT-base-uncased + LoRA adapters")
    print("Task: Binary classification (Disaster vs Non-disaster)")
    print()
    print("Processing modes:")
    print("1. Batch processing - Process entire CSV at once")
    print("2. Streaming simulation - Process CSV in small batches with delays")
    print()
    
    # Choose processing mode
    mode = input("Choose mode (1: Batch processing, 2: Streaming simulation): ").strip()
    
    if mode == "2":
        process_streaming_csv()
    else:
        process_csv_stream()
