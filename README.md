# BERT Disaster Prediction with Kafka and Spark

This project uses your trained BERT model to predict whether text messages describe disasters. It includes both Kafka streaming and Spark processing capabilities.

## 📋 Prerequisites

- Python 3.8+ 
- Java 8 or 11 (required for Kafka and Spark)
- Apache Kafka 2.8.0+
- Apache Spark 3.4.0+
- Windows PowerShell or Command Prompt

## 🛠️ Installation

### 1. Install Python Dependencies

```powershell
pip install torch transformers peft pandas pyspark kafka-python
```

### 2. Download and Setup Kafka

1. **Download Kafka:**
   - Go to https://kafka.apache.org/downloads
   - Download the latest Kafka binary (e.g., `kafka_2.13-2.8.0.tgz`)
   
2. **Extract Kafka:**
   ```powershell
   # Extract to C:\kafka (or your preferred location)
   # Your folder structure should be: C:\kafka\bin\windows\
   ```

3. **Set Environment Variables (Optional):**
   ```powershell
   # Add to your PATH
   $env:PATH += ";C:\kafka\bin\windows"
   ```

### 3. Download and Setup Spark

1. **Download Spark:**
   - Go to https://spark.apache.org/downloads.html
   - Download Spark 3.4.0+ pre-built for Hadoop
   
2. **Extract Spark:**
   ```powershell
   # Extract to C:\spark (or your preferred location)
   ```

3. **Set SPARK_HOME:**
   ```powershell
   $env:SPARK_HOME = "C:\spark"
   $env:PATH += ";C:\spark\bin"
   ```

## 🚀 Quick Start Guide

### Step 1: Start Zookeeper

Zookeeper manages Kafka cluster coordination.

```powershell
# Navigate to Kafka directory
cd C:\kafka

# Start Zookeeper (keep this terminal open)
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

**Expected Output:**
```
[2024-01-01 10:00:00,000] INFO binding to port 0.0.0.0/0.0.0.0:2181
```

### Step 2: Start Kafka Server

In a **new terminal**:

```powershell
# Navigate to Kafka directory
cd C:\kafka

# Start Kafka server (keep this terminal open)
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

**Expected Output:**
```
[2024-01-01 10:00:30,000] INFO [KafkaServer id=0] started
```

### Step 3: Create Kafka Topics

In a **new terminal**:

```powershell
cd C:\kafka

# Create topic for text input
.\bin\windows\kafka-topics.bat --create --topic disaster-text --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Create topic for predictions output
.\bin\windows\kafka-topics.bat --create --topic disaster-predictions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# List topics to verify
.\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```

**Expected Output:**
```
disaster-predictions
disaster-text
```

### Step 4: Run the Disaster Prediction System

Now you can run your BERT prediction system:

```powershell
# Navigate to your project directory
cd "C:\Users\ADMIN''\PycharmProjects\big-data-project"

# Option 1: Run with Spark (fixed version)
python spark_fixed.py

# Option 2: Run simple pandas version
python working_bert_predictor.py

# Option 3: Test your model first
python quick_test.py
```

## 📊 Kafka Streaming Setup (Optional)

If you want to create a real-time streaming pipeline:

### 1. Create Kafka Producer Script

```python
# kafka_producer.py
from kafka import KafkaProducer
import json
import pandas as pd
import time

def send_text_data():
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Read your CSV data
    df = pd.read_csv("data/test.csv")
    
    for index, row in df.iterrows():
        if pd.notna(row['text']):
            message = {
                'id': index,
                'text': row['text'],
                'timestamp': time.time()
            }
            
            producer.send('disaster-text', value=message)
            print(f"Sent: {row['text'][:50]}...")
            time.sleep(1)  # Send one message per second
    
    producer.close()

if __name__ == "__main__":
    send_text_data()
```

### 2. Create Kafka Consumer Script

```python
# kafka_consumer.py
from kafka import KafkaConsumer
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def load_bert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert_model")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased", num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, "bert_model")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def consume_and_predict():
    # Load model
    model, tokenizer, device = load_bert_model()
    
    # Create consumer
    consumer = KafkaConsumer(
        'disaster-text',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    print("🔄 Listening for messages...")
    
    for message in consumer:
        data = message.value
        text = data['text']
        
        # Predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()
        
        result = "DISASTER" if pred == 1 else "SAFE"
        print(f"📝 {result} ({conf:.3f}): {text[:60]}...")

if __name__ == "__main__":
    consume_and_predict()
```

### 3. Run the Streaming Pipeline

In **separate terminals**:

```powershell
# Terminal 1: Start producer
python kafka_producer.py

# Terminal 2: Start consumer
python kafka_consumer.py
```

## 🔧 Troubleshooting

### Common Issues

**1. "Java not found" error:**
```powershell
# Install Java and set JAVA_HOME
$env:JAVA_HOME = "C:\Program Files\Java\jdk-11.0.x"
```

**2. Kafka connection refused:**
```powershell
# Check if Zookeeper is running first
# Then start Kafka server
# Verify with: netstat -an | findstr 9092
```

**3. Python worker crashes in Spark:**
```powershell
# Use the fixed version:
python spark_fixed.py
# Or use the simple pandas version:
python working_bert_predictor.py
```

**4. BERT model loading issues:**
```powershell
# Test your model first:
python quick_test.py
```

### Useful Kafka Commands

```powershell
# List all topics
.\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092

# Delete a topic
.\bin\windows\kafka-topics.bat --delete --topic disaster-text --bootstrap-server localhost:9092

# Describe a topic
.\bin\windows\kafka-topics.bat --describe --topic disaster-text --bootstrap-server localhost:9092

# Monitor messages in a topic
.\bin\windows\kafka-console-consumer.bat --topic disaster-text --from-beginning --bootstrap-server localhost:9092
```

### Useful Spark Commands

```powershell
# Start Spark shell
spark-shell

# Submit a Spark application
spark-submit --class YourClass your-app.jar

# Check Spark UI (when running)
# Open browser: http://localhost:4040
```

## 📁 Project Structure

```
big-data-project/
├── bert_model/              # Your trained BERT model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── data/
│   └── test.csv            # Input data
├── output/
│   └── predictions/        # Results
├── spark_fixed.py          # Fixed Spark version
├── working_bert_predictor.py # Simple pandas version
├── quick_test.py           # Model testing
└── README.md              # This file
```

## 🎯 Performance Tips

1. **For large datasets:** Use `spark_fixed.py`
2. **For quick testing:** Use `working_bert_predictor.py`
3. **For streaming:** Set up Kafka pipeline
4. **Memory issues:** Reduce batch size in scripts
5. **GPU acceleration:** Ensure CUDA is available

## 📈 Monitoring

- **Kafka UI:** http://localhost:9092 (if running Kafka Manager)
- **Spark UI:** http://localhost:4040 (when Spark job is running)
- **Check logs:** Look at terminal outputs for errors

## 🛑 Shutdown Sequence

When done, shut down in this order:

```powershell
# 1. Stop your Python applications (Ctrl+C)
# 2. Stop Kafka server (Ctrl+C in Kafka terminal)
# 3. Stop Zookeeper (Ctrl+C in Zookeeper terminal)
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify Java installation
3. Ensure all services are running in order
4. Test with `quick_test.py` first

---

**Happy Disaster Prediction! 🚨📊**
- PySpark 3.5.6 with Scala 2.12
- Python 3.9

## Getting Started

### 1. Start the services

```bash
docker compose up -d
```

This will:

- Start Zookeeper
- Start Kafka
- Create the required Kafka topic "test-tweets"
- Start a PySpark container

### 2. Run the Kafka producer

Connect to the PySpark container and run the producer:

```bash
docker exec -it pyspark bash
DOCKER_ENV=true python /app/kafka_stream_docker.py
```

This will start streaming data from the CSV file into Kafka.

### 3. Run the Spark consumer

In a separate terminal, connect to the PySpark container and run the Spark job:

```bash
docker exec -it pyspark bash
$SPARK_HOME/bin/spark-submit --master local[*] /app/spark_stream_docker.py
```

This will read from Kafka and write the results to both console and files in the `/app/output/stream` directory.

### 4. Check the output

The output files will be available in the `output/stream` directory in your project folder.

## Stopping the services

```bash
docker compose down
```

## Troubleshooting

If you encounter issues:

1. Check the Kafka logs:

```bash
docker logs kafka
```

2. Verify the topic exists:

```bash
docker exec -it kafka kafka-topics --bootstrap-server kafka:29092 --list
```

3. Check the data in the topic:

```bash
docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic test-tweets --from-beginning
```

4. Check the Spark application logs:

```bash
docker logs pyspark
```
