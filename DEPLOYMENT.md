# Deployment Guide - Audio Classifier Demo

This guide provides step-by-step instructions for deploying the Audio Classifier Demo application.

## 🚀 Quick Deployment

### Option 1: Local Development (Recommended for testing)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python run.py
   # or
   python app.py
   ```

3. **Access the application**
   ```
   http://localhost:5000
   ```

### Option 2: Production with Gunicorn

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

3. **Access the application**
   ```
   http://localhost:8000
   ```

## 🌐 Cloud Deployment

### Heroku Deployment

1. **Create Heroku app**
   ```bash
   heroku create your-audio-classifier-app
   ```

2. **Create Procfile**
   ```bash
   echo "web: gunicorn app:app" > Procfile
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

4. **Open the app**
   ```bash
   heroku open
   ```

### Railway Deployment

1. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will auto-detect Python and deploy

2. **Set environment variables**
   ```bash
   FLASK_ENV=production
   FLASK_DEBUG=0
   ```

### Render Deployment

1. **Create Render account**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository

2. **Configure service**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Port**: `8000`

## 🐳 Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Build and Run
```bash
# Build the image
docker build -t audio-classifier-demo .

# Run the container
docker run -p 5000:5000 audio-classifier-demo
```

## 🔧 Environment Configuration

### Development Environment
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export SECRET_KEY=dev-secret-key
export MAX_FILE_SIZE=16777216
```

### Production Environment
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export SECRET_KEY=your-production-secret-key
export MAX_FILE_SIZE=16777216
export HOST=0.0.0.0
export PORT=5000
```

## 📊 Performance Optimization

### For High Traffic

1. **Use multiple workers**
   ```bash
   gunicorn -w 8 -b 0.0.0.0:8000 app:app
   ```

2. **Enable compression**
   ```bash
   pip install flask-compress
   ```

3. **Add caching**
   ```bash
   pip install flask-caching
   ```

### Memory Optimization

1. **Reduce batch size** in `app.py`
2. **Use smaller audio files** (max 10MB)
3. **Enable garbage collection**

## 🔒 Security Considerations

### Production Checklist

- [ ] Change default `SECRET_KEY`
- [ ] Enable HTTPS
- [ ] Set up proper CORS headers
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Set up logging
- [ ] Configure error handling

### Security Headers
```python
# Add to app.py
from flask_talisman import Talisman

# Enable HTTPS
Talisman(app, force_https=True)
```

## 📝 Monitoring and Logging

### Add logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/audio_classifier.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Audio Classifier startup')
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill the process
   kill -9 <PID>
   ```

2. **Memory issues**
   ```bash
   # Monitor memory usage
   htop
   # Restart with more memory
   gunicorn -w 2 --max-requests 1000 app:app
   ```

3. **Audio processing errors**
   - Check file format support
   - Verify librosa installation
   - Test with smaller files

### Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

## 📈 Scaling

### Horizontal Scaling
- Use load balancer (nginx)
- Deploy multiple instances
- Use Redis for session storage

### Vertical Scaling
- Increase server resources
- Optimize model inference
- Use GPU acceleration

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
        heroku_app_name: "your-app-name"
        heroku_email: "your-email@example.com"
```

## 📞 Support

For deployment issues:
1. Check the logs: `heroku logs --tail`
2. Verify environment variables
3. Test locally first
4. Check the troubleshooting section

---

**Note**: Always test your deployment in a staging environment before going to production. 