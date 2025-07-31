class FlagGuessingGame {
    constructor() {
        this.currentCountry = null;
        this.currentFlag = null;
        this.score = 0;
        this.round = 1;
        this.totalRounds = 0;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordedBlob = null;
        this.isRecording = false;

        this.initializeElements();
        this.initializeEventListeners();
        this.startNewRound();
    }

    initializeElements() {
        // UI Elements
        this.flagEmoji = document.getElementById('flag-emoji');
        this.scoreDisplay = document.getElementById('score');
        this.roundDisplay = document.getElementById('round');
        this.accuracyDisplay = document.getElementById('accuracy');
        
        // Recording elements
        this.recordBtn = document.getElementById('record-btn');
        this.submitBtn = document.getElementById('submit-btn');
        this.reRecordBtn = document.getElementById('re-record-btn');
        this.recordingIndicator = document.getElementById('recording-indicator');
        this.audioPreview = document.getElementById('audio-preview');
        this.audioPlayback = document.getElementById('audio-playback');
        
        // Result elements
        this.resultDisplay = document.getElementById('result-display');
        this.predictionText = document.getElementById('prediction-text');
        this.confidenceBadge = document.getElementById('confidence-badge');
        this.correctnessDisplay = document.getElementById('correctness');
        this.correctAnswerDisplay = document.getElementById('correct-answer');
        this.topPredictionsDisplay = document.getElementById('top-predictions');
        this.nextRoundBtn = document.getElementById('next-round-btn');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loading-overlay');
    }

    initializeEventListeners() {
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.submitBtn.addEventListener('click', () => this.submitGuess());
        this.reRecordBtn.addEventListener('click', () => this.reRecord());
        this.nextRoundBtn.addEventListener('click', () => this.startNewRound());
        this.flagEmoji.addEventListener('click', () => this.showFlagAnimation());
    }

    async startNewRound() {
        try {
            // Reset UI
            this.resetUI();
            
            // Get new random country
            const response = await fetch('/get_random_country');
            const data = await response.json();
            
            this.currentCountry = data.country;
            this.currentFlag = data.flag;
            
            // Update display
            this.flagEmoji.textContent = this.currentFlag;
            this.roundDisplay.textContent = this.round;
            
            console.log(`New round: ${this.currentCountry} ${this.currentFlag}`);
        } catch (error) {
            console.error('Error starting new round:', error);
            this.showError('Failed to load new country. Please try again.');
        }
    }

    resetUI() {
        // Hide result display
        this.resultDisplay.classList.add('hidden');
        
        // Reset recording state
        this.audioPreview.classList.add('hidden');
        this.recordingIndicator.classList.add('hidden');
        this.submitBtn.disabled = true;
        this.recordBtn.disabled = false;
        this.recordBtn.classList.remove('recording');
        
        // Reset recording button text
        this.recordBtn.querySelector('.record-text').textContent = 'Start Recording';
        
        // Clear recorded audio
        this.recordedBlob = null;
        this.audioChunks = [];
    }

    showFlagAnimation() {
        this.flagEmoji.style.transform = 'scale(1.2) rotate(10deg)';
        setTimeout(() => {
            this.flagEmoji.style.transform = 'scale(1)';
        }, 300);
    }

    async toggleRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 22050,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.recordedBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.showAudioPreview();
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            this.recordBtn.classList.add('recording');
            this.recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
            this.recordingIndicator.classList.remove('hidden');
            this.audioPreview.classList.add('hidden');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Failed to access microphone. Please check permissions.');
        }
    }

    async stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            this.recordBtn.classList.remove('recording');
            this.recordBtn.querySelector('.record-text').textContent = 'Start Recording';
            this.recordingIndicator.classList.add('hidden');
        }
    }

    showAudioPreview() {
        const audioUrl = URL.createObjectURL(this.recordedBlob);
        this.audioPlayback.src = audioUrl;
        this.audioPreview.classList.remove('hidden');
        this.submitBtn.disabled = false;
    }

    reRecord() {
        this.resetUI();
        URL.revokeObjectURL(this.audioPlayback.src);
    }

    async submitGuess() {
        if (!this.recordedBlob) {
            this.showError('No audio recorded. Please record your guess first.');
            return;
        }

        try {
            this.showLoading(true);
            
            const formData = new FormData();
            formData.append('audio', this.recordedBlob, 'audio.webm');
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            this.showResults(result);
            
        } catch (error) {
            console.error('Error submitting guess:', error);
            this.showError(`Failed to process audio: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    showResults(result) {
        const predictedCountry = result.predicted_country;
        const confidence = result.confidence;
        const isCorrect = predictedCountry.toLowerCase() === this.currentCountry.toLowerCase();
        
        // Update score
        if (isCorrect) {
            this.score++;
        }
        this.totalRounds++;
        
        // Update displays
        this.scoreDisplay.textContent = this.score;
        this.accuracyDisplay.textContent = `${Math.round((this.score / this.totalRounds) * 100)}%`;
        
        // Show prediction
        this.predictionText.textContent = predictedCountry;
        this.confidenceBadge.textContent = `${Math.round(confidence * 100)}% confident`;
        
        // Show correctness
        this.correctnessDisplay.textContent = isCorrect ? '✅ Correct!' : '❌ Incorrect';
        this.correctnessDisplay.className = `correctness ${isCorrect ? 'correct' : 'incorrect'}`;
        
        // Show correct answer
        this.correctAnswerDisplay.textContent = this.currentCountry;
        
        // Show top predictions
        this.showTopPredictions(result.all_predictions);
        
        // Show result display
        this.resultDisplay.classList.remove('hidden');
        
        // Update round for next time
        this.round++;
    }

    showTopPredictions(allPredictions) {
        const sortedPredictions = Object.entries(allPredictions)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
        
        const html = `
            <h4>🔍 Top 5 AI Predictions:</h4>
            ${sortedPredictions.map(([country, confidence]) => `
                <div class="prediction-item">
                    <span>${country}</span>
                    <span>${Math.round(confidence * 100)}%</span>
                </div>
            `).join('')}
        `;
        
        this.topPredictionsDisplay.innerHTML = html;
    }

    showLoading(show) {
        if (show) {
            this.loadingOverlay.classList.remove('hidden');
        } else {
            this.loadingOverlay.classList.add('hidden');
        }
    }

    showError(message) {
        alert(`Error: ${message}`);
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check for required browser features
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Sorry, your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Safari.');
        return;
    }
    
    // Initialize the game
    new FlagGuessingGame();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, you might want to pause recording
        console.log('Page hidden');
    } else {
        // Page is visible again
        console.log('Page visible');
    }
});