# Financial RAG Chatbot Deployment Guide

This guide describes how to deploy the Financial RAG Chatbot to various platforms so you can access it via a generic URL instead of a Replit URL.

## Option 1: Deploy with Streamlit Cloud

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub/GitLab repository containing this app
3. Configure the deployment settings:
   - Set the main file path to `main.py`
   - Add any required secrets in the settings panel

Streamlit Cloud will host your app and provide a public URL, with options to connect a custom domain.

## Option 2: Deploy to Heroku

1. Install the Heroku CLI and create an account
2. Create a `Procfile` in your project root with this content:
   ```
   web: streamlit run main.py --server.port=$PORT
   ```
3. Initialize a git repository if you haven't already:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   ```
4. Deploy to Heroku:
   ```
   heroku create your-app-name
   git push heroku main
   ```

Your app will be available at `https://your-app-name.herokuapp.com`.

## Option 3: Deploy to AWS Elastic Beanstalk

1. Install the [AWS CLI](https://aws.amazon.com/cli/) and [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)
2. Create a `Procfile` in your project root:
   ```
   web: streamlit run main.py --server.address=0.0.0.0 --server.port=5000
   ```
3. Create a `.ebextensions/01_packages.config` file:
   ```yaml
   packages:
     yum:
       gcc: []
   ```
4. Initialize and deploy:
   ```
   eb init -p python-3.8 financial-rag-chatbot
   eb create financial-rag-environment
   ```

## Option 4: Self-Host on VPS or Dedicated Server

### Setup on Ubuntu Server:

1. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev build-essential
   ```

2. Clone your repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

3. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Run with Gunicorn (for production):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:80 --worker-class=gthread --threads=4 -k uvicorn.workers.UvicornWorker streamlit_app:app
   ```

5. Set up Nginx as a reverse proxy (optional but recommended):
   ```bash
   sudo apt install nginx
   ```

   Create a new Nginx configuration:
   ```bash
   sudo nano /etc/nginx/sites-available/streamlit
   ```

   Add:
   ```
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable the site:
   ```bash
   sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

6. Set up SSL with Let's Encrypt (optional):
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

## Docker Deployment

1. Create a `Dockerfile`:
   ```Dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 5000

   CMD ["streamlit", "run", "main.py", "--server.port=5000", "--server.address=0.0.0.0"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t financial-rag-chatbot .
   docker run -p 80:5000 financial-rag-chatbot
   ```

## Important Notes for Deployment

1. Make sure to handle environment variables for sensitive information
2. Consider the resource requirements - this app uses ML models that may need significant memory
3. For production use, you might want to use a more scalable vector database than FAISS

## Creating a requirements.txt file

To generate a requirements.txt file for deployment, run:

```bash
pip freeze > requirements.txt
```

Or create one manually with these key dependencies:

```
streamlit>=1.26.0
langchain>=0.0.267
langchain-community>=0.0.10
torch>=2.0.0
transformers>=4.30.2
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pypdf>=3.15.1
nltk>=3.8.1
rank_bm25>=0.2.2
```

## Need Help?

If you encounter issues with your deployment, consider:
1. Checking platform-specific documentation
2. Reviewing logs for error messages
3. Consulting with a DevOps specialist if deploying to production environments