# 🚀 Deployment Guide

This guide covers deploying your Flag Guessing Game to various cloud platforms.

## ⚠️ Before Deploying

1. **Add your model files** to the `models/` directory:
   - `audio_country_enhanced_transformer_model.keras`
   - `label_encoder_enhanced_transformer.pkl`

2. **Test locally** using `./run.sh` or `python app.py`

3. **Commit your code** (but not the model files - they're gitignored)

---

## 🟢 Railway (Recommended - Easiest)

Railway is perfect for this project and offers generous free tier.

### Steps:
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** and select your repository
3. **Add model files** via Railway's file manager or CLI
4. **Deploy** - Railway auto-detects Flask apps!

### Commands:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up

# Upload model files
railway shell
# Then upload your model files to the models/ directory
```

**✅ Pros**: Free tier, auto-deployment, easy file uploads  
**❌ Cons**: File upload can be slow for large models

---

## 🔵 Heroku

Classic platform with good Flask support.

### Steps:
1. **Install Heroku CLI**: Download from [heroku.com/cli](https://devcenter.heroku.com/articles/heroku-cli)
2. **Create app**:
   ```bash
   heroku create your-flag-game-name
   ```
3. **Add model files** using Heroku's large file support:
   ```bash
   # Install git-lfs for large files
   git lfs track "models/*.keras"
   git lfs track "models/*.pkl"
   git add .gitattributes
   git add models/
   git commit -m "Add model files"
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```

**✅ Pros**: Reliable, well-documented  
**❌ Cons**: Paid plans required for larger apps, file size limits

---

## 🟠 Google Cloud Run

Serverless container platform, scales to zero.

### Steps:
1. **Install Google Cloud CLI**: [cloud.google.com/cli](https://cloud.google.com/sdk/docs/install)
2. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```
3. **Build and deploy**:
   ```bash
   # Set your project ID
   export PROJECT_ID=your-project-id
   
   # Build container
   gcloud builds submit --tag gcr.io/$PROJECT_ID/flag-game
   
   # Deploy to Cloud Run
   gcloud run deploy flag-game \
     --image gcr.io/$PROJECT_ID/flag-game \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2
   ```

**✅ Pros**: Scales to zero, pay per use, powerful  
**❌ Cons**: More complex setup, requires Google Cloud account

---

## 🟡 AWS App Runner

Simple containerized deployment on AWS.

### Steps:
1. **Push to ECR**:
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name flag-game
   
   # Get login token
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin \
     <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and push
   docker build -t flag-game .
   docker tag flag-game:latest \
     <account-id>.dkr.ecr.us-east-1.amazonaws.com/flag-game:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/flag-game:latest
   ```

2. **Create App Runner service** via AWS Console or CLI

**✅ Pros**: Simple AWS deployment, auto-scaling  
**❌ Cons**: Requires AWS account, more expensive than Railway

---

## 🔴 DigitalOcean App Platform

Container-based platform with simple deployment.

### Steps:
1. **Connect GitHub** at [cloud.digitalocean.com/apps](https://cloud.digitalocean.com/apps)
2. **Select repository** and configure:
   - **Build Command**: `docker build -t flag-game .`
   - **Run Command**: `gunicorn --bind 0.0.0.0:8080 app:app`
   - **Environment**: Production
   - **Instance Size**: Basic ($5/month minimum)
3. **Upload model files** via SSH or file manager

**✅ Pros**: Simple setup, good docs  
**❌ Cons**: Paid plans only, requires manual file upload

---

## 🐳 Docker Deployment (Any Platform)

Use Docker on any cloud platform that supports containers.

### Build and Test Locally:
```bash
# Build image
docker build -t flag-game .

# Run locally
docker run -p 5000:5000 -v $(pwd)/models:/app/models flag-game

# Test
curl http://localhost:5000/health
```

### Deploy to any Docker platform:
- **Fly.io**: `fly deploy`
- **Render**: Connect GitHub, auto-deploys
- **Azure Container Instances**: `az container create`

---

## 📊 Performance Considerations

### Memory Requirements:
- **Minimum**: 1GB RAM (model loading)
- **Recommended**: 2GB RAM (for better performance)
- **Storage**: ~500MB (model files + dependencies)

### Optimization Tips:
1. **Model Loading**: Consider lazy loading for faster startup
2. **Caching**: Cache model predictions for common inputs
3. **Audio Processing**: Limit file sizes (already implemented)
4. **Scaling**: Use multiple workers for high traffic

### Environment Variables:
```bash
# Optional configurations
FLASK_ENV=production
MODEL_PATH=/app/models/
MAX_CONTENT_LENGTH=16777216  # 16MB
```

---

## 🔐 Security Checklist

- [ ] Model files are not in Git repository
- [ ] HTTPS enabled (required for microphone access)
- [ ] File upload limits configured
- [ ] Error handling doesn't expose sensitive info
- [ ] Health checks implemented

---

## 🐛 Common Issues

### "Model not found" errors:
- Ensure model files are in the correct directory
- Check file permissions and names
- Verify files weren't corrupted during upload

### Audio recording not working:
- **HTTPS required** for microphone access in production
- Check browser compatibility
- Verify microphone permissions

### Memory errors:
- Increase container memory allocation
- Consider model optimization
- Check for memory leaks in audio processing

### Slow predictions:
- Use GPU-enabled instances if available
- Consider model quantization
- Implement prediction caching

---

## 📈 Monitoring

Set up monitoring for:
- **Health endpoint**: `/health`
- **Response times**: Track prediction latency
- **Error rates**: Monitor failed predictions
- **Resource usage**: Memory and CPU usage

### Example monitoring URLs:
- Health: `https://your-app.com/health`
- Metrics: Use platform-specific monitoring tools

---

## 🎯 Success Checklist

- [ ] Application loads without errors
- [ ] Health endpoint returns "healthy"
- [ ] Flags display correctly
- [ ] Audio recording works (HTTPS required)
- [ ] AI predictions return results
- [ ] Score tracking functions
- [ ] Mobile-responsive design works
- [ ] Game flow is smooth

**🎉 Congratulations! Your AI-powered Flag Guessing Game is live!**