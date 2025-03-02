import nltk
import os

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Add to NLTK's data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download necessary NLTK data - punkt_tab removed as it doesn't exist
        resources = ["punkt", "stopwords"]
        for resource in resources:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {e}")
        
        print("NLTK resources downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download NLTK resources: {e}")
        return False

# If this script is run directly, download the resources
if __name__ == "__main__":
    download_nltk_resources()