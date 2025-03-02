import os
import subprocess
import sys
import importlib.util

# Check if we're running in Streamlit Cloud or Replit
is_streamlit_cloud = os.environ.get('IS_STREAMLIT_CLOUD') == 'true'

if is_streamlit_cloud:
    # For Streamlit Cloud - use default port 8501
    os.environ['PORT'] = '8501'
else:
    # For Replit - use port 5000
    os.environ['PORT'] = '5000'

# Call the download_nltk script to ensure resources are available
try:
    # Execute the download_nltk.py script directly
    print("Downloading NLTK resources...")
    subprocess.run([sys.executable, "download_nltk.py"], check=False)
    print("NLTK resources download process completed")
except Exception as e:
    print(f"Warning: Failed to run NLTK resource download script: {e}")

# Run the main Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])