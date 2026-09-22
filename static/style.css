let chart;

async function analyzeTraffic() {
    const input = document.getElementById('featureInput').value;
    const features = input.split(',').map(Number);

    if (features.length < 80) {
        alert("Please provide at least 80 features.");
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: features })
        });

        const data = await response.json();

        if (data.status === "success") {
            displayResults(data);
        }
    } catch (error) {
        console.error("Error:", error);
    }
}

function displayResults(data) {
    document.getElementById('resultPlaceholder').classList.add('hidden');
    document.getElementById('resultContent').classList.remove('hidden');

    const label = document.getElementById('predictionLabel');
    label.innerText = data.prediction;
    
    // Color coding based on threat level
    label.className = data.prediction === 'Normal' ? 'text-4xl font-black text-green-400' : 'text-4xl font-black text-red-500';
    
    document.getElementById('confidenceLabel').innerText = (data.confidence * 100).toFixed(2) + "%";

    updateChart(data.confidence);
}

function updateChart(confidence) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Ensemble Confidence'],
            datasets: [{
                data: [confidence],
                backgroundColor: '#3b82f6',
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            scales: { x: { min: 0, max: 1, display: false }, y: { display: false } },
            plugins: { legend: { display: false } }
        }
    });
}