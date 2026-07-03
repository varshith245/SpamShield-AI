document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const messageInput = document.getElementById('message-input');
    const charCount = document.getElementById('char-count');
    const clearBtn = document.getElementById('clear-btn');
    const scanBtn = document.getElementById('scan-btn');
    const scanAgainBtn = document.getElementById('scan-again-btn');
    const inputCard = document.querySelector('.input-card');
    const resultsCard = document.getElementById('results-card');
    const modelAccuracyDisplay = document.getElementById('model-accuracy');

    // Fetch Stats
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (data.accuracy) {
                modelAccuracyDisplay.textContent = (data.accuracy * 100).toFixed(1) + '%';
            }
        })
        .catch(err => console.error('Error fetching stats:', err));

    // Character Count
    messageInput.addEventListener('input', () => {
        charCount.textContent = `${messageInput.value.length} chars`;
    });

    // Clear Input
    clearBtn.addEventListener('click', () => {
        messageInput.value = '';
        charCount.textContent = '0 chars';
    });

    // Scan Logic
    scanBtn.addEventListener('click', async () => {
        const text = messageInput.value.trim();
        if (!text) {
            alert('Please enter some text to scan.');
            return;
        }

        const formData = new FormData();
        formData.append('text', text);

        // UI Loading State
        const btnText = scanBtn.querySelector('.btn-text');
        const loader = scanBtn.querySelector('.loader');
        
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        scanBtn.disabled = true;

        try {
            const response = await fetch('/scan', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showResults(result);
            } else {
                alert(result.error || 'An error occurred during scanning.');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Network error. Please try again.');
        } finally {
            btnText.style.display = 'inline';
            loader.style.display = 'none';
            scanBtn.disabled = false;
        }
    });

    function showResults(data) {
        inputCard.style.display = 'none';
        resultsCard.style.display = 'block';
        
        const verdictLabel = document.getElementById('verdict-label');
        const verdictIcon = document.getElementById('verdict-icon');
        const confidenceText = document.getElementById('confidence-text');
        const confidenceCircle = document.getElementById('confidence-circle');
        const confidencePercentage = document.getElementById('confidence-percentage');
        const keywordsList = document.getElementById('keywords-list');
        const messagePreview = document.getElementById('message-preview');

        // Update Verdict
        verdictLabel.textContent = data.verdict;
        if (data.verdict === 'SPAM') {
            verdictLabel.style.color = 'var(--danger-color)';
            verdictIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color: var(--danger-color)"></i>';
            confidenceCircle.style.stroke = 'var(--danger-color)';
        } else {
            verdictLabel.style.color = 'var(--success-color)';
            verdictIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color: var(--success-color)"></i>';
            confidenceCircle.style.stroke = 'var(--success-color)';
        }

        // Update Confidence
        confidenceText.textContent = `${data.confidence}% Confidence`;
        confidencePercentage.textContent = `${Math.round(data.confidence)}%`;
        
        // Animate Circle
        // stroke-dasharray: value, 100
        confidenceCircle.style.strokeDasharray = `${data.confidence}, 100`;

        // Update Keywords
        keywordsList.innerHTML = '';
        if (data.keywords && data.keywords.length > 0) {
            data.keywords.forEach(word => {
                const tag = document.createElement('span');
                tag.className = 'keyword-tag';
                tag.textContent = word;
                keywordsList.appendChild(tag);
            });
        } else {
            keywordsList.innerHTML = '<span style="color: var(--text-muted)">No specific spam keywords detected.</span>';
        }

        // Update Preview
        messagePreview.textContent = `"${data.message_preview}"`;
    }

    // Scan Again
    scanAgainBtn.addEventListener('click', () => {
        resultsCard.style.display = 'none';
        inputCard.style.display = 'block';
    });
});
