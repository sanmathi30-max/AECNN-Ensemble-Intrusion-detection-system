import numpy as np
import csv
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sample_traffic(label="Normal"):
    """
    Generates a synthetic 80-feature vector with randomized threat signatures.
    Ensures the AE-CNN 'spatial' detection is tested against varying positions.
    """
    # Start with base ambient noise (the 'Normal' background)
    base = np.random.uniform(0.01, 0.12, 80) 
    
    if label == "DoS":
        # Randomize the start point of the DoS burst (within the first half)
        start = random.randint(0, 10)
        length = random.randint(15, 25)
        # High intensity saturation with jitter
        base[start:start+length] = np.random.uniform(0.8, 1.0, length) 
    
    elif label == "Probe":
        # Randomize the 'stripe' frequency to simulate different scanning tools
        step = random.choice([3, 4, 5])
        offset = random.randint(0, 2)
        for i in range(offset, 80, step):
            base[i] = np.random.uniform(0.85, 0.95)
            
    elif label == "R2L":
        # Shift the 'payload island' randomly around the center
        center_shift = random.randint(25, 40)
        size = random.randint(12, 18)
        base[center_shift:center_shift+size] = np.random.normal(0.7, 0.1, size)
        
    elif label == "U2R":
        # Rare, subtle spikes at random locations (mimicking privilege escalation)
        for _ in range(random.randint(3, 6)):
            idx = random.randint(0, 79)
            base[idx] = np.random.uniform(0.6, 0.9)
        
    # Final clipping to keep values within Neural Network range [0, 1]
    return np.clip(base, 0, 1).tolist()

# Define a more diverse dataset mix
dataset_config = {
    "Normal": 150,
    "DoS": 45,
    "Probe": 35,
    "R2L": 15,
    "U2R": 10
}

filename = "traffic_test.csv"
all_rows = []

print(f"--- 🛡️  Generating Randomized AE-CNN Dataset ---")

for label, count in dataset_config.items():
    for _ in range(count):
        row = generate_sample_traffic(label)
        all_rows.append(row)

# Shuffle to ensure the dashboard charts update dynamically
random.shuffle(all_rows)

try:
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_rows)
    print(f"✅ Created {filename} with {len(all_rows)} rows of randomized traffic.")
    print("🚀 Ready to test Ensemble Voting robustness.")
except Exception as e:
    print(f"❌ Error: {e}")