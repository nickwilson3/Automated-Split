import os

# Base directory of the application
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Upload folder for Excel files
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Results folder
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Maximum upload file size (5MB)
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}