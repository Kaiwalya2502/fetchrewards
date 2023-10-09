window.onload = getOriginalData;

async function getOriginalData() {
    // Fetch original data and plot
    const response = await fetch('/get_original_data');
    const data = await response.json();
    plotGraph(data.x, data.y, 'Original Data');
}

async function getPrediction() {
    const days = document.getElementById('daysInput').value;
    // Fetch predicted data and plot
    const response = await fetch('/predict?days=' + days);
    const data = await response.json();
    plotGraph(data.x, data.y, 'Prediction', data.total);
}

function plotGraph(x, y, name, total = 0) {
    const trace = {
        type: 'scatter',
        mode: 'lines',
        name: name,
        x: x,
        y: y,
        line: {color: name === 'Original Data' ? 'read' : 'orange'}
    };
    Plotly.newPlot('plotlyGraph', [trace], {title: 'Receipt Count Prediction'});
    document.getElementById('total').innerText = total;
}
