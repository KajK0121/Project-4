// Function to make a prediction using the machine learning model
function predict() {
    // Fetch input values from the form
    const gender = document.getElementById('gender').value;
    const age = parseFloat(document.getElementById('age').value);
    const bmi = parseFloat(document.getElementById('bmi').value);
    const hba1c = parseFloat(document.getElementById('hba1c').value);
    const bloodGlucose = parseFloat(document.getElementById('bloodGlucose').value);

    // Make a request to your Flask backend for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            gender: gender,
            age: age,
            bmi: bmi,
            hba1c: hba1c,
            bloodGlucose: bloodGlucose,
        }),
    })
    .then(response => response.json())
    .then(prediction => {
        // Update the UI with the prediction result
        document.getElementById('predictionResult').innerHTML = `<p>Prediction Result: ${prediction.result}</p>`;

        // You can add further visualization or UI updates here based on the prediction
        // For example, call a function to update a Plotly chart
        updatePlotlyChart(prediction);
    })
    .catch(error => console.error('Error:', error));
}
// Function to update Plotly chart based on prediction
function updatePlotlyChart(prediction) {
    // Sample data for illustration purposes
    const chartData = [{
        x: ['Male', 'Female'],
        y: [10, 20],  // Replace with actual data from your Flask backend
        type: 'bar',
        marker: {
            color: ['#FF69B4', '#FFC0CB']  // Pink colors for Male and Female
        }
    }];

    const layout = {
        title: 'Gender Distribution',
        xaxis: { title: 'Gender' },
        yaxis: { title: 'Count' }
    };

    // Update or create the Plotly chart
    Plotly.newPlot('plotlyChart', chartData, layout);
}
