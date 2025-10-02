document.getElementById('sentimentFormV2').addEventListener('submit', async function (e) {
    e.preventDefault();

    const textInput = document.getElementById('textInputV2');
    const loading = document.getElementById('loadingV2');
    const results = document.getElementById('resultsV2');
    const error = document.getElementById('errorV2');

    // Reset displays
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    try {
        const response = await fetch('/v2/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: textInput.value
            })
        });

        const data = await response.json();

        loading.style.display = 'none';

        if (data.error) {
            error.textContent = data.error;
            error.style.display = 'block';
        } else {
            displayResultsV2(data);
        }

    } catch (err) {
        loading.style.display = 'none';
        error.textContent = 'An error occurred while analyzing the sentiment.';
        error.style.display = 'block';
    }
});

function displayResultsV2(data) {
    const results = document.getElementById('resultsV2');
    const sentimentIcon = document.getElementById('sentimentIconV2');
    const sentimentLabel = document.getElementById('sentimentLabelV2');
    const sentimentScore = document.getElementById('sentimentScoreV2');
    const confidenceBar = document.getElementById('confidenceBarV2');

    // Set icon and colors based on sentiment
    let icon, colorClass, bgClass;

    switch (data.label.toUpperCase()) {
        case 'POSITIVE':
            icon = '<i class="fas fa-smile sentiment-icon sentiment-positive"></i>';
            colorClass = 'sentiment-positive';
            bgClass = 'bg-success';
            break;
        case 'NEGATIVE':
            icon = '<i class="fas fa-frown sentiment-icon sentiment-negative"></i>';
            colorClass = 'sentiment-negative';
            bgClass = 'bg-danger';
            break;
        case 'NEUTRAL':
        default:
            icon = '<i class="fas fa-meh sentiment-icon sentiment-neutral"></i>';
            colorClass = 'sentiment-neutral';
            bgClass = 'bg-warning';
            break;
    }

    sentimentIcon.innerHTML = icon;
    sentimentLabel.textContent = data.label;
    sentimentLabel.className = colorClass;
    sentimentScore.textContent = `Score: ${data.score.toFixed(4)}`;

    // Update confidence bar
    confidenceBar.style.width = `${data.confidence}%`;
    confidenceBar.className = `progress-bar ${bgClass}`;
    confidenceBar.textContent = `${data.confidence.toFixed(1)}%`;

    // Show results with animation
    results.style.display = 'block';
    results.classList.add('fade-in');

    // Remove animation class after animation completes
    setTimeout(() => {
        results.classList.remove('fade-in');
    }, 500);
}
