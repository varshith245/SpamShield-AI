# Deploying SpamShield AI to Vercel

This guide will walk you through deploying your Flask application to Vercel.

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
   ```bash
   npm install -g vercel
   ```
3. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, or Bitbucket)

## ⚠️ Important Considerations

### Vercel Limitations for Flask Apps

1. **Serverless Functions**: Vercel runs Flask as serverless functions (not a traditional server)
2. **File Uploads**: Ephemeral storage - uploaded files are temporary
3. **Model Files**: Must be included in deployment (can increase size)
4. **Cold Starts**: First request may be slower (model loading)
5. **Memory Limits**: Free tier has 1GB RAM limit

### Recommended Alternatives

For production Flask apps, consider:
- **Railway** (Recommended) - Easy Flask deployment
- **Render** - Free tier available
- **Fly.io** - Good for Python apps
- **Heroku** - Classic option (paid)
- **DigitalOcean App Platform** - Production-ready

## 🚀 Deployment Steps

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Configure Project

The project already includes `vercel.json` configuration. Review it:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

### Step 4: Update .gitignore (if needed)

Ensure these are NOT in .gitignore (Vercel needs them):
- `models/` - Model files must be included
- `templates/` - HTML templates
- `static/` - CSS/JS files

### Step 5: Deploy

#### Option A: Deploy via CLI

```bash
# From project root
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (Your account)
# - Link to existing project? No
# - Project name? spam-shield-ai (or your choice)
# - Directory? ./
```

#### Option B: Deploy via GitHub Integration

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "Add New Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `./`
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)
5. Click "Deploy"

### Step 6: Environment Variables (if needed)

If you need environment variables:

```bash
# Via CLI
vercel env add VARIABLE_NAME

# Or via Dashboard
# Settings → Environment Variables
```

## 📁 Project Structure for Vercel

```
spam-ham/
├── api/
│   └── index.py          # Vercel serverless function entry point
├── app.py                 # Flask application
├── vercel.json           # Vercel configuration
├── requirements.txt       # Python dependencies
├── models/               # Model files (must be included)
├── templates/            # HTML templates
├── static/               # Static assets
└── ...
```

## 🔧 Configuration Files

### vercel.json

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

**Note**: The `functions` property cannot be used with `builds`. If you need to configure function settings like `maxDuration`, set them in the Vercel dashboard under Project Settings → Functions.

### api/index.py

This file serves as the entry point for Vercel's serverless functions.

## 🐛 Troubleshooting

### Issue: Model Not Loading

**Solution**: Ensure model files are committed to Git and not in .gitignore

```bash
# Check if models are tracked
git ls-files models/

# If not, add them
git add models/
git commit -m "Add model files for deployment"
```

### Issue: Import Errors

**Solution**: Ensure all dependencies are in `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Issue: Static Files Not Loading

**Solution**: Check `vercel.json` routes configuration

### Issue: Cold Start Timeout

**Solution**: Increase function timeout in Vercel Dashboard:
1. Go to Project Settings → Functions
2. Set "Max Duration" to 60 seconds (or higher)
3. Or use Vercel CLI:
   ```bash
   vercel env add VERCEL_FUNCTION_MAX_DURATION
   # Enter value: 60
   ```

### Issue: Memory Limit Exceeded

**Solution**: 
- Optimize model size
- Use model quantization
- Consider upgrading Vercel plan

## 📊 Monitoring

### View Logs

```bash
# Via CLI
vercel logs

# Or via Dashboard
# Project → Logs
```

### View Deployments

```bash
vercel ls
```

## 🔄 Updating Deployment

### Update and Redeploy

```bash
# Make changes
git add .
git commit -m "Update app"
git push

# Redeploy
vercel --prod
```

Or push to main branch (if GitHub integration is set up)

## 🌐 Custom Domain

1. Go to Project Settings → Domains
2. Add your domain
3. Follow DNS configuration instructions

## 💰 Pricing

### Free Tier Limits:
- 100GB bandwidth/month
- 100 serverless function executions/day
- 1GB memory per function
- 10s execution time (can be increased to 60s)

### Pro Tier ($20/month):
- Unlimited bandwidth
- Unlimited executions
- 3GB memory
- 300s execution time

## 📝 Best Practices

1. **Optimize Model Size**: Consider model compression
2. **Cache Model**: Load model once, reuse across requests
3. **Error Handling**: Add proper error handling for serverless
4. **Logging**: Use Vercel's logging for debugging
5. **Environment Variables**: Store secrets in Vercel env vars

## 🔗 Alternative Deployment Options

### Railway (Recommended for Flask)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up
```

### Render

1. Connect GitHub repository
2. Select "Web Service"
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Launch
fly launch

# Deploy
fly deploy
```

## 📚 Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- [Vercel CLI Reference](https://vercel.com/docs/cli)
- [Flask on Vercel Guide](https://vercel.com/guides/deploying-flask-with-vercel)

## ✅ Deployment Checklist

- [ ] Vercel account created
- [ ] Vercel CLI installed
- [ ] `vercel.json` configured
- [ ] `api/index.py` created
- [ ] Model files committed to Git
- [ ] `requirements.txt` up to date
- [ ] Static files properly configured
- [ ] Environment variables set (if needed)
- [ ] Tested locally with `vercel dev`
- [ ] Deployed to production
- [ ] Custom domain configured (optional)

---

**Note**: For production use, consider Railway, Render, or Fly.io as they're better suited for Flask applications with ML models.

