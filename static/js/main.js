// Main JavaScript for Audio Classifier Demo

// Global variables
let currentAudioFile = null;
let predictionHistory = [];

// Utility functions
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateAudioFile(file) {
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/m4a', 'audio/flac'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: 'Invalid file type. Please upload WAV, MP3, M4A, or FLAC files.' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, error: 'File size too large. Maximum size is 16MB.' };
    }
    
    return { valid: true };
}

function updateFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    if (fileInfo) {
        fileInfo.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-music me-2"></i>
                <div>
                    <strong>${file.name}</strong><br>
                    <small class="text-muted">${formatFileSize(file.size)}</small>
                </div>
            </div>
        `;
    }
}

function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="text-center text-white">
            <div class="loading-spinner mb-3"></div>
            <h5>Processing Audio...</h5>
            <p>Please wait while we analyze your audio file</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function hideLoadingOverlay() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

function animateProgressBar(progressBar, targetWidth, duration = 1000) {
    const startWidth = 0;
    const startTime = performance.now();
    
    function updateProgress(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentWidth = startWidth + (targetWidth - startWidth) * progress;
        progressBar.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(updateProgress);
        }
    }
    
    requestAnimationFrame(updateProgress);
}

function createConfidenceChart(containerId, predictions) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const ctx = document.createElement('canvas');
    container.appendChild(ctx);
    
    const labels = predictions.map(p => p.country);
    const data = predictions.map(p => p.confidence);
    const colors = ['#28a745', '#ffc107', '#dc3545', '#6c757d', '#17a2b8'];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, predictions.length),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });
}

function saveToPredictionHistory(prediction) {
    predictionHistory.push({
        ...prediction,
        timestamp: new Date().toISOString(),
        id: Date.now()
    });
    
    // Keep only last 10 predictions
    if (predictionHistory.length > 10) {
        predictionHistory = predictionHistory.slice(-10);
    }
    
    localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
}

function loadPredictionHistory() {
    const saved = localStorage.getItem('predictionHistory');
    if (saved) {
        predictionHistory = JSON.parse(saved);
    }
}

function displayPredictionHistory() {
    const historyContainer = document.getElementById('predictionHistory');
    if (!historyContainer || predictionHistory.length === 0) return;
    
    let html = '<h5>Recent Predictions</h5><div class="row">';
    
    predictionHistory.slice(-6).reverse().forEach(pred => {
        const date = new Date(pred.timestamp).toLocaleDateString();
        html += `
            <div class="col-md-6 mb-2">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">${pred.country}</h6>
                        <div class="progress mb-2">
                            <div class="progress-bar" style="width: ${pred.confidence}%"></div>
                        </div>
                        <small class="text-muted">${pred.confidence}% confidence - ${date}</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    historyContainer.innerHTML = html;
}

// Audio recording utilities
function initializeAudioRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showNotification('Audio recording is not supported in this browser.', 'warning');
        return false;
    }
    return true;
}

function createAudioVisualizer(audioElement, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaElementSource(audioElement);
    
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        const width = canvas.width;
        const height = canvas.height;
        
        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgb(0, 0, 0)';
        ctx.fillRect(0, 0, width, height);
        
        const barWidth = (width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;
            
            const gradient = ctx.createLinearGradient(0, 0, 0, height);
            gradient.addColorStop(0, '#007bff');
            gradient.addColorStop(1, '#28a745');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
    }
    
    draw();
}

// File upload enhancements
function enhanceFileUpload() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput) return;
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const validation = validateAudioFile(file);
        if (!validation.valid) {
            showNotification(validation.error, 'danger');
            fileInput.value = '';
            return;
        }
        
        currentAudioFile = file;
        updateFileInfo(file);
        showNotification('File selected successfully!', 'success');
    });
}

// Performance monitoring
function measurePerformance(operation, callback) {
    const startTime = performance.now();
    
    return function(...args) {
        const result = callback.apply(this, args);
        const endTime = performance.now();
        
        console.log(`${operation} took ${(endTime - startTime).toFixed(2)}ms`);
        return result;
    };
}

// Error handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    let userMessage = 'An unexpected error occurred. Please try again.';
    
    if (error.name === 'NetworkError') {
        userMessage = 'Network error. Please check your connection and try again.';
    } else if (error.name === 'QuotaExceededError') {
        userMessage = 'File size too large. Please choose a smaller file.';
    } else if (error.message.includes('audio')) {
        userMessage = 'Audio processing error. Please try a different file.';
    }
    
    showNotification(userMessage, 'danger');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load prediction history
    loadPredictionHistory();
    
    // Enhance file upload
    enhanceFileUpload();
    
    // Initialize audio recording if supported
    if (initializeAudioRecording()) {
        console.log('Audio recording initialized');
    }
    
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add fade-in animation to cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const submitBtn = document.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
        
        // Escape to close modals/alerts
        if (e.key === 'Escape') {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            });
        }
    });
    
    // Add service worker for offline support (if supported)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('ServiceWorker registration successful');
            })
            .catch(error => {
                console.log('ServiceWorker registration failed:', error);
            });
    }
});

// Export functions for use in templates
window.AudioClassifierDemo = {
    showNotification,
    formatFileSize,
    validateAudioFile,
    showLoadingOverlay,
    hideLoadingOverlay,
    animateProgressBar,
    createConfidenceChart,
    saveToPredictionHistory,
    displayPredictionHistory,
    createAudioVisualizer,
    handleError
}; 