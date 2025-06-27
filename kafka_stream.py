import json
import csv
import time
from kafka import KafkaProducer
import os

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8")
)

folder = "data"
topic = "test-tweets"
csv_file = os.path.join(folder, "test.csv")

# Process the CSV file
with open(csv_file, "r", encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        try:
            # Send each row as a message
            producer.send(topic, value=row)
            # Print confirmation with the text field
            print("✅ Sent:", row.get("text", "")[:60], "...")
            # Small delay to simulate streaming
            time.sleep(0.1)
        except Exception as e:
            print("❌ Error:", e)

# Make sure all messages are sent
producer.flush()
