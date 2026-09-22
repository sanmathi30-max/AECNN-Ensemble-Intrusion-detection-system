import io
import csv
import torch
import torch.nn as nn
import datetime  # Added missing import
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE (AE-CNN Ensemble)
# ---------------------------------------------------------
# 1. Define the Architecture (Must match your training script)
class AE_CNN_Model(nn.Module):
    def __init__(self):
        super(AE_CNN_Model, self).__init__()
        # Encoder: 80 -> 16 (The 4x4 Latent Space)
        self.encoder = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 16) 
        )
        # Decoder: 16 -> 80 (For Reconstruction Error/MSE)
        self.decoder = nn.Sequential(
            nn.Linear(16, 40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.Sigmoid()
        )
        # CNN Ensemble: Processes the 4x4 Grid
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 5) # 5 Classes: Normal, DoS, etc.
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # Spatial transformation for CNN
        spatial = latent.view(-1, 1, 4, 4)
        logits = self.cnn(spatial)
        return latent, reconstructed, logits

# 2. Initialize the "names" so Flask can see them
model = AE_CNN_Model()
model.eval() # Set to evaluation mode

# 3. Create aliases so your upload code works
def autoencoder(input_tensor):
    latent, reconstructed, _ = model(input_tensor)
    return latent, reconstructed

def cnn_ensemble(spatial_grid):
    # This expects the 4x4 grid already reshaped
    _, _, logits = model(torch.zeros(1, 80)) # Dummy call logic
    # In your route, just call: _, _, prediction = model(input_tensor)
    return logits

CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
analysis_history = [] # Global history store

# ---------------------------------------------------------
# 2. ROUTES & CONTROLLERS
# ---------------------------------------------------------

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global analysis_history
    
    # 1. Calculate Total Scanned
    total_count = len(analysis_history)
    
    # 2. Calculate Total Threats (Excluding 'Normal')
    # Use a list comprehension to strictly filter out 'Normal'
    threat_list = [x for x in analysis_history if x['type'] != 'Normal']
    threat_count = len(threat_list)
    
    # 3. Dynamic Model Confidence
    # Extracts the float value from the 'conf' string (e.g., "98.5%" -> 98.5)
    if total_count > 0:
        conf_values = [float(x['conf'].strip('%')) for x in analysis_history]
        avg_confidence = sum(conf_values) / total_count
    else:
        avg_confidence = 0

    # 4. Helper for Chart Percentages
    def get_pct(label):
        if total_count == 0: return 0
        count = len([x for x in analysis_history if x['type'] == label])
        return (count / total_count) * 100

    stats = {
        "total": "{:,}".format(total_count),
        "threats": "{:,}".format(threat_count),
        "avg_conf": f"{avg_confidence:.2f}%",
        "normal_pct": get_pct('Normal'),
        "dos_pct": get_pct('DoS'),
        "probe_pct": get_pct('Probe'),
        "special_pct": get_pct('R2L') + get_pct('U2R')
    }
    
    return render_template('dashboard.html', stats=stats)
@app.route('/upload', methods=['POST'])
def upload_file():
    global analysis_history
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    
    try:
        # Read the file stream
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        reader = csv.reader(stream)
        
        batch_results = []
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        for row in reader:
            if not row or len(row) < 80: continue
            
            features = torch.FloatTensor([float(x) for x in row[:80]]).view(1, -1)

            with torch.no_grad():
                # Single pass through our integrated AE-CNN model
                latent, reconstructed, prediction = model(features)
                
                # Calculate MSE (Reconstruction Loss)
                mse_loss = torch.mean((features - reconstructed)**2).item()
                
                # Calculate Confidence and Label
                prob = torch.softmax(prediction, dim=1)
                conf_score = torch.max(prob).item()
                class_idx = torch.argmax(prob, dim=1).item()
                
                res = {
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": CLASSES[class_idx],
                    "conf": f"{conf_score:.2%}",
                    "mse": f"{mse_loss:.5f}"
                }
                analysis_history.insert(0, res)
                batch_results.append(res)
            
        return jsonify({"status": "success", "count": len(batch_results)})

    except Exception as e:
        # This prints the REAL error to your Python terminal
        print(f"❌ SERVER ERROR: {str(e)}") 
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/reports')
def reports():
    global analysis_history
    total = len(analysis_history)
    
    # Data for the 4 visualizations
    types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    counts = [len([x for x in analysis_history if x['type'] == t]) for t in types]
    
    # Calculate Average Confidence
    avg_conf = sum([float(x['conf'].strip('%')) for x in analysis_history]) / total if total > 0 else 0
    
    return render_template('reports.html', 
                           history=analysis_history, 
                           labels=types, 
                           values=counts,
                           avg_conf=round(avg_conf, 2), zip=zip)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('features')
    if not data or len(data) < 80:
        return jsonify({"status": "error", "message": "Invalid input"}), 400
    
    input_tensor = torch.FloatTensor(data).view(1, -1)
    with torch.no_grad():
        prediction = model(input_tensor)
        idx = torch.argmax(prediction, dim=1).item()
        
    return jsonify({
        "status": "success",
        "prediction": CLASSES[idx],
        "confidence": float(torch.max(prediction).item())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)