# integrated_app.py (No Changes Needed from Previous Version with Chart Data)
import os
import pandas as pd
import pickle
import PyPDF2 # For reading PDFs
import docx # For reading DOCX
from flask import (
    Flask, request, render_template, send_from_directory,
    redirect, url_for, jsonify, flash, send_file, session # Added session
)
from werkzeug.utils import secure_filename
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer # Keep for type hints if needed
from sklearn.metrics.pairwise import cosine_similarity # For comparison
import zipfile # For downloading logs by category
import io # For creating zip in memory
import json # <-- Keep this for potential future use or other JSON needs
import numpy as np # For keyword extraction helper (might still be needed)
import logging
import uuid # For unique filenames
from datetime import datetime # For footer year

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- End Logging Setup ---


# --- NLTK Resource Check (Ensure data exists) ---
# Use LookupError for NLTK's specific exception
resources = ['punkt', 'stopwords', 'wordnet']
nltk_data_verified = True
for resource in resources:
    try:
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}')
        elif resource == 'stopwords':
             nltk.data.find(f'corpora/{resource}.zip') # Check for zip file
        elif resource == 'wordnet':
             nltk.data.find(f'corpora/{resource}.zip') # Check for zip file
        else:
             nltk.data.find(f'corpora/{resource}') # General case
        logging.info(f"NLTK resource '{resource}' found.")
    except LookupError: # Correct NLTK exception
        logging.warning(f"NLTK resource '{resource}' not found. Attempting download...")
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"NLTK resource '{resource}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Error downloading NLTK resource '{resource}': {e}. Functionality might be limited.")
            nltk_data_verified = False # Mark as failed
# --- End NLTK Check ---

app = Flask(__name__)

# --- Configuration ---
# IMPORTANT: Change this secret key in a production environment!
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-replace-this')
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true' # Set via env var for production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Or 'Strict' if appropriate

# Define Folders relative to the app's root path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")           # For classifier resume storage & logs
COMPARE_FOLDER = os.path.join(APP_ROOT, "uploads_compare_temp") # Temp storage for comparison resumes
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPARE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPARE_FOLDER'] = COMPARE_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# --- Load Model Artifacts ---
# Define paths relative to the app's root path
MODEL_PATH = os.path.join(APP_ROOT, 'best_model.pkl')
VECTORIZER_PATH = os.path.join(APP_ROOT, 'tfidf_vectorizer.pkl')
ENCODER_PATH = os.path.join(APP_ROOT, 'label_encoder.pkl')
ACCURACY_PATH = os.path.join(APP_ROOT, 'model_accuracy.txt')

model, vectorizer, label_encoder, accuracy = None, None, None, "N/A"
model_loaded, vectorizer_loaded, encoder_loaded = False, False, False

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f); model_loaded = True
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f); vectorizer_loaded = True
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f: label_encoder = pickle.load(f); encoder_loaded = True
    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH, "r") as f: accuracy = f.read().strip()
    else:
         accuracy = "Not Found"

    logging.info(f"Classifier Model Loaded: {model_loaded}")
    logging.info(f"Vectorizer Loaded: {vectorizer_loaded}")
    logging.info(f"Label Encoder Loaded: {encoder_loaded}")
    logging.info(f"Accuracy Loaded: {accuracy}")

    if not all([model_loaded, vectorizer_loaded, encoder_loaded]):
        logging.warning("Warning: Not all classifier components loaded. Classification might fail.")
    if not vectorizer_loaded:
         logging.critical("CRITICAL WARNING: Vectorizer not loaded. Both classification and comparison will fail.")
    if not nltk_data_verified:
         logging.warning("Warning: Some NLTK data downloads failed or were missing. Text processing might be incomplete.")

except FileNotFoundError as fnf_error:
    logging.error(f"❌ Error loading artifacts: File not found - {fnf_error}. Ensure model files exist in the same directory as the app or provide correct paths.")
except Exception as e:
    logging.error(f"❌ Error loading one or more model artifacts: {e}", exc_info=True)
# --- End Loading ---

# --- Logs for Classifier ---
log_file = os.path.join(UPLOAD_FOLDER, "logs.csv") # Log file inside uploads folder
LOG_COLUMNS = ["filename", "predicted_role", "job_title"]
if not os.path.exists(log_file):
    try:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(log_file, index=False)
        logging.info(f"Created log file: {log_file}")
    except Exception as e:
        logging.error(f"Error creating log file {log_file}: {e}")
# --- End Logs ---

# --- Text Preprocessing Setup ---
stop_words = set()
lemmatizer = None
preprocessing_ready = False
if nltk_data_verified:
    try:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        # Test lemmatizer
        _ = lemmatizer.lemmatize("tests")
        preprocessing_ready = True
        logging.info("NLTK stop words and lemmatizer loaded successfully.")
    except LookupError as le:
        logging.error(f"Failed to load NLTK resource for preprocessing: {le}. Text cleaning will be basic.")
    except Exception as e:
        logging.error(f"Unexpected error loading NLTK resources: {e}")

# Ensure these phrases match the ones removed during training (if any)
common_phrases_to_remove = [
    "curriculum vitae", "resume", "objective", "declaration",
    "personal information", "contact", "address", "phone", "email",
    "date of birth", "dob", "nationality", "gender", "marital status",
    "personal details", "summary", "profile", "career objective", "references"
]

def clean_text(text, for_tfidf=True):
    """Cleans text. Set for_tfidf=False for basic cleaning without stopword/lemma/phrase removal."""
    if not isinstance(text, str): return ""
    text_lower = text.lower()

    # Basic cleaning applied regardless of for_tfidf
    text_cleaned = re.sub(r'<.*?>', ' ', text_lower) # Remove HTML, replace with space
    text_cleaned = re.sub(r'http\S+|www\S+', ' ', text_cleaned) # Remove URLs
    text_cleaned = re.sub(r'\S+@\S+', ' ', text_cleaned) # Remove email addresses
    # Keep letters, numbers, and essential symbols like + # . (e.g., C++, C#, .NET). Remove others.
    text_cleaned = re.sub(r'[^\w\s\+\#\.]', ' ', text_cleaned)
    text_cleaned = re.sub(r'\b\d+\b', ' ', text_cleaned) # Remove standalone numbers
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip() # Normalize whitespace

    if not for_tfidf or not preprocessing_ready: # If basic cleaning or NLTK failed
        return text_cleaned # Return early

    # Full cleaning for TF-IDF continues only if NLTK data is ready
    # Remove common phrases using regex word boundaries
    current_text = text_cleaned
    for phrase in common_phrases_to_remove:
        try:
            # Use \b for word boundaries, make case insensitive
            current_text = re.sub(r'\b' + re.escape(phrase) + r'\b', ' ', current_text, flags=re.IGNORECASE)
        except re.error as re_err:
             logging.warning(f"Regex error removing phrase '{phrase}': {re_err}")
             continue

    text_cleaned = re.sub(r'\s+', ' ', current_text).strip() # Re-normalize whitespace

    if not text_cleaned:
        return ""

    try:
        tokens = word_tokenize(text_cleaned)
        # Lemmatize AND filter non-alpha (or keep alpha-numeric if needed)
        # Keep words longer than 2 chars, not stopwords
        if lemmatizer:
            filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens
                               if word.isalnum() and len(word) > 2 and word not in stop_words] # isalnum allows numbers within words like C++
        else: # Fallback if lemmatizer failed
            filtered_tokens = [word for word in tokens
                               if word.isalnum() and len(word) > 2 and word not in stop_words]

        return " ".join(filtered_tokens)
    except Exception as e:
         logging.warning(f"Error tokenizing/lemmatizing in clean_text: {e}. Text: '{text_cleaned[:50]}...'")
         # Return the partially cleaned text instead of empty string
         return re.sub(r'\s+', ' ', text_cleaned).strip()


def extract_text_from_file(file_path, filename):
    """Extracts text from PDF, DOCX, or TXT files. Returns text or error string."""
    text = ""
    if not os.path.exists(file_path):
        logging.warning(f"File not found during extraction: {file_path}")
        return f"Error: File '{filename}' not found at expected location."

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    try:
        if ext == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    # Use strict=False for potentially corrupted PDFs
                    reader = PyPDF2.PdfReader(f, strict=False)
                    if reader.is_encrypted:
                        try:
                            # Try decrypting with an empty password
                            decrypt_status = reader.decrypt('')
                            if decrypt_status == PyPDF2.PasswordType.OWNER_PASSWORD or decrypt_status == PyPDF2.PasswordType.USER_PASSWORD:
                                logging.info(f"Decrypted PDF: {filename}")
                            elif decrypt_status == PyPDF2.PasswordType.NOT_DECRYPTED:
                                logging.warning(f"Could not decrypt PDF {filename} (password protected).")
                                return "Error: PDF is password protected and could not be decrypted."
                        except Exception as decrypt_err:
                             logging.warning(f"Error trying to decrypt PDF {filename}: {decrypt_err}")
                             # Continue assuming it might not actually be encrypted or readable anyway

                    page_texts = []
                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                page_texts.append(page_text)
                        except Exception as page_error:
                            # Log page-specific errors but continue if possible
                            logging.warning(f"Error extracting text from page {i+1} of {filename}: {page_error}")
                    text = " ".join(page_texts)
                    if not text.strip() and len(reader.pages) > 0:
                        logging.warning(f"PDF '{filename}' contained pages but no text could be extracted (possibly image-based).")
                        # text = "Info: PDF contains pages but no extractable text (likely image-based)."

            except PyPDF2.errors.PdfReadError as pdf_err:
                 logging.error(f"PyPDF2 error reading {filename}: {pdf_err}")
                 return f"Error: Could not read PDF file. It might be corrupted or incompatible. ({pdf_err})"
            except Exception as e:
                 logging.error(f"General error reading PDF {filename}: {e}", exc_info=True)
                 return f"Error: Unexpected issue processing PDF file. Check logs for details."

        elif ext == ".docx":
             try:
                doc = docx.Document(file_path)
                # Join paragraphs with newline, filter empty ones
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
             except Exception as e:
                 logging.error(f"Error reading DOCX {filename}: {e}", exc_info=True)
                 return f"Error: Failed to process DOCX file. It might be corrupted. Check logs."

        elif ext == ".txt":
            # Try common encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        text = f.read()
                    # logging.info(f"Read TXT {filename} with encoding {enc}")
                    read_success = True
                    break # Stop trying if successful
                except UnicodeDecodeError:
                    continue # Try next encoding
                except Exception as e:
                     logging.error(f"Error reading TXT {filename} with {enc}: {e}")
                     text = f"Error: Failed to read TXT file. Check logs." # Set error text
                     break # Stop trying encodings on other errors

            if not read_success and not text.startswith("Error:"):
                 logging.warning(f"Could not read TXT file {filename} with attempted encodings.")
                 text = "Error: Unknown text encoding or read error for TXT file."

        else:
             logging.warning(f"Unsupported file type for {filename}: {ext}")
             text = f"Error: Unsupported file type '{ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"

    except FileNotFoundError:
         logging.error(f"File disappeared during extraction: {filename}")
         return "Error: File not found during processing."
    except Exception as e:
        logging.error(f"Unexpected error processing file {filename} during text extraction: {e}", exc_info=True)
        return "Error: Unexpected file processing failure. Check logs."

    if isinstance(text, str) and not text.strip() and not text.startswith("Error:") and not text.startswith("Info:"):
        logging.warning(f"File '{filename}' was read successfully but contained no text content.")
        # text = "Info: File read successfully but is empty."

    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_job_title(text):
    """Attempts to extract a job title from raw text using keywords. More robust version."""
    if not isinstance(text, str) or not text.strip(): return "Not Found"
    if text.startswith("Error:") or text.startswith("Info:"): return "Extraction Failed/Info"

    search_text = text[:1000].lower() # Limit search scope

    # Prioritize longer, more specific titles first
    tech_keywords = [ # Simplified list for example
        "software development engineer", "software engineer", "machine learning engineer", "data scientist",
        "web developer", "full stack developer", "devops engineer", "systems administrator", "it specialist",
    ]
    business_management_keywords = [ # Simplified list for example
        "project manager", "product manager", "business analyst", "data analyst", "it manager",
    ]
    hr_finance_other_keywords = [ # Simplified list for example
        "hr manager", "human resources specialist", "recruiter", "accountant", "financial analyst",
    ]
    all_keywords = tech_keywords + business_management_keywords + hr_finance_other_keywords

    for keyword in all_keywords:
        try:
            if re.search(r'\b' + re.escape(keyword) + r'\b', search_text, re.IGNORECASE):
                return keyword.title()
        except re.error as re_err:
             logging.warning(f"Regex error searching for job title keyword '{keyword}': {re_err}")
             continue

    # Fallback general roles
    if "developer" in search_text: return "Developer (General)"
    if "engineer" in search_text: return "Engineer (General)"
    if "analyst" in search_text: return "Analyst (General)"
    if "manager" in search_text: return "Manager (General)"
    # ... other fallbacks ...

    logging.info(f"Could not find a specific job title keyword.")
    return "Not Found"


def get_matching_keywords(cleaned_resume_text, cleaned_job_role_text, resume_vector, vectorizer_obj, top_n=15):
    """Finds keywords from the resume that are also in the job role, ranked by TF-IDF score."""
    if resume_vector is None or vectorizer_obj is None or not hasattr(vectorizer_obj, 'get_feature_names_out'):
        logging.warning("Cannot get keywords, vector or vectorizer invalid for keyword extraction.")
        return [], 0
    # ... (rest of the keyword extraction logic remains the same) ...
    if not hasattr(resume_vector, 'indices'):
         logging.warning("Resume vector does not have 'indices' attribute (might not be sparse CSR). Keyword extraction might fail.")

    try:
        job_role_tokens = set(cleaned_job_role_text.split())
        if not job_role_tokens:
            logging.warning("Cleaned job role text is empty, cannot find matching keywords.")
            return [], 0

        feature_names = vectorizer_obj.get_feature_names_out()
        word_scores = {}

        if hasattr(resume_vector, 'indices') and hasattr(resume_vector, 'data') and len(resume_vector.indices) == len(resume_vector.data):
             indices = resume_vector.indices
             data = resume_vector.data
             max_feature_index = len(feature_names)
             for idx, score in zip(indices, data):
                 if score > 0.01 and idx < max_feature_index:
                     word = feature_names[idx]
                     if word in job_role_tokens:
                         word_scores[word] = score
                 elif idx >= max_feature_index:
                      logging.warning(f"Index {idx} out of bounds for feature_names (len {max_feature_index}).")
        else:
            # Fallback logic...
            logging.warning("Resume vector is not in expected sparse format. Using less efficient keyword extraction.")
            if hasattr(resume_vector, 'toarray'):
                dense_vector = resume_vector.toarray()[0]
                for i, score in enumerate(dense_vector):
                     if score > 0.01 and i < len(feature_names):
                         word = feature_names[i]
                         if word in job_role_tokens:
                             word_scores[word] = score
                     elif i >= len(feature_names):
                          logging.warning(f"Dense Index {i} out of bounds for feature_names (len {len(feature_names)}).")
            else:
                 logging.error("Cannot process resume vector for keywords - unexpected format.")
                 return ["Error: Vector format issue"], 0

        if not word_scores:
            logging.info("No common keywords found between resume and job role.")
            return [], 0

        sorted_keywords = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
        formatted_keywords = [f"{word} ({score:.2f})" for word, score in sorted_keywords[:top_n]]
        total_matched_count = len(word_scores)
        return formatted_keywords, total_matched_count

    except Exception as e:
        logging.error(f"Unexpected error getting matching keywords: {e}", exc_info=True)
        return ["Error: Unexpected issue retrieving keywords"], 0


def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Add context processor for current year ---
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}

# --- Flask Routes ---

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serves files from the main UPLOAD_FOLDER."""
    if '..' in filename or filename.startswith('/'):
        logging.warning(f"Attempt to access invalid path rejected: {filename}")
        flash("Invalid file path requested.", "danger")
        return redirect(url_for('index'))

    safe_filename = secure_filename(filename)
    if not safe_filename:
         logging.warning(f"Filename became empty after securing: {filename}")
         flash("Invalid filename requested.", "danger")
         return redirect(url_for('index'))

    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename, as_attachment=False)
    except FileNotFoundError:
         logging.warning(f"File not found in uploads folder: {safe_filename}")
         flash(f"File '{safe_filename}' not found. It might have been cleared.", "warning")
         return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error serving file {safe_filename} from {app.config['UPLOAD_FOLDER']}: {e}", exc_info=True)
        flash("Error serving file.", "danger")
        return redirect(url_for('index'))


# Main route for classifier and dashboard
@app.route("/", methods=["GET", "POST"])
def index():
    """Handles classifier form submission and displays dashboard + results."""

    # --- Classification POST Request Logic ---
    if request.method == "POST" and 'resumes' in request.files:
        classifier_ready = all([model_loaded, vectorizer_loaded, encoder_loaded])
        if not classifier_ready:
            flash("Classifier components are not fully loaded. Cannot predict categories.", "danger")
            return redirect(url_for('index'))
        if not preprocessing_ready:
            flash("NLTK preprocessing components failed to load. Text cleaning will be basic; classification accuracy may be affected.", "warning")

        files = request.files.getlist("resumes")
        current_batch_results = []
        current_batch_pdfs = []
        new_log_entries = []
        processed_count = 0
        error_count = 0
        skipped_count = 0

        # Load existing log data safely
        df_log_existing = pd.DataFrame(columns=LOG_COLUMNS)
        if os.path.exists(log_file):
            try:
                df_log_existing = pd.read_csv(log_file)
                if not all(col in df_log_existing.columns for col in LOG_COLUMNS):
                     logging.warning(f"Log file {log_file} has incorrect columns. Resetting log.")
                     df_log_existing = pd.DataFrame(columns=LOG_COLUMNS)
            except pd.errors.EmptyDataError:
                logging.info(f"Log file {log_file} is empty.")
            except Exception as e:
                 logging.error(f"Error reading classification log file '{log_file}': {e}", exc_info=True)
                 flash("Error reading existing log file.", "warning")

        if not files or all(f.filename == '' for f in files):
             flash("No files selected for classification.", "warning")
        else:
            for file in files:
                original_filename = file.filename
                if not file or not original_filename:
                    skipped_count += 1
                    continue

                if allowed_file(original_filename):
                    display_name = original_filename
                    filename_base = secure_filename(original_filename)
                    if not filename_base:
                         logging.warning(f"Skipping invalid filename: {original_filename}")
                         skipped_count += 1
                         continue

                    unique_suffix = uuid.uuid4().hex[:8]
                    name, ext = os.path.splitext(filename_base)
                    unique_filename = f"{name[:50]}_{unique_suffix}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                    try:
                        file.save(file_path)
                        logging.info(f"Saved: '{original_filename}' as '{unique_filename}'")
                        processed_count += 1
                    except Exception as e:
                        logging.error(f"Error saving {unique_filename}: {e}", exc_info=True)
                        flash(f"Error saving file: {display_name}.", "danger")
                        error_count += 1
                        continue

                    raw_text = extract_text_from_file(file_path, unique_filename)
                    predicted_category = "Processing Error"; job_title_extracted = "N/A"

                    if isinstance(raw_text, str) and raw_text.startswith("Error:"):
                         predicted_category = "Extraction Error"; job_title_extracted = "Extraction Failed/Info"
                         logging.warning(f"Extraction failed for {unique_filename}: {raw_text}")
                         error_count +=1
                    else:
                        cleaned_text_tfidf = clean_text(raw_text, for_tfidf=True)
                        job_title_extracted = extract_job_title(raw_text)
                        if not cleaned_text_tfidf or not cleaned_text_tfidf.strip():
                            predicted_category = "Unreadable/Empty"
                        else:
                            try:
                                vector = vectorizer.transform([cleaned_text_tfidf])
                                if hasattr(vector, 'nnz') and vector.nnz == 0:
                                    predicted_category = "No Features Found"
                                else:
                                    prediction_encoded = model.predict(vector)[0]
                                    predicted_category = label_encoder.inverse_transform([prediction_encoded])[0]
                                    logging.info(f"Predicted '{predicted_category}' for {unique_filename}")
                            except Exception as e:
                                logging.error(f"Prediction error for {unique_filename}: {e}", exc_info=True)
                                predicted_category = "Prediction Error"
                                error_count += 1

                    file_url = url_for('uploaded_file', filename=unique_filename, _external=False)
                    result_item = {
                        "display_filename": display_name, "stored_filename": unique_filename,
                        "prediction": predicted_category, "job_title": job_title_extracted, "file_url": file_url
                    }
                    current_batch_results.append(result_item)

                    if unique_filename.lower().endswith('.pdf'):
                         current_batch_pdfs.append({"url": file_url, "name": display_name})

                    loggable_categories = not predicted_category.lower().startswith("error") and \
                                          predicted_category not in ["Unreadable/Empty", "No Features Found"]
                    if loggable_categories:
                        new_log_entries.append({
                            "filename": unique_filename, "predicted_role": predicted_category, "job_title": job_title_extracted
                        })
                    else:
                         logging.info(f"Skipping logging for {unique_filename} (Status: {predicted_category})")

                elif original_filename:
                     flash(f"File type not allowed: '{original_filename}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "warning")
                     skipped_count += 1

            # Update log file
            if new_log_entries:
                df_new_entries = pd.DataFrame(new_log_entries)
                try:
                    df_log_updated = pd.concat([df_log_existing, df_new_entries], ignore_index=True)
                    df_log_updated = df_log_updated.drop_duplicates(subset=['filename'], keep='last')
                    df_log_updated.to_csv(log_file, index=False)
                    logging.info(f"Appended {len(new_log_entries)} entries to log file: {log_file}")
                except Exception as e:
                    logging.error(f"Error writing updates to log file '{log_file}': {e}", exc_info=True)
                    flash("Error updating classification log file.", "warning")

            # Flash summary
            summary_message = f"Processed {processed_count} file(s)."
            if new_log_entries: summary_message += f" {len(new_log_entries)} added to history."
            if error_count: summary_message += f" {error_count} encountered errors."
            if skipped_count: summary_message += f" {skipped_count} skipped."
            flash(summary_message, "info")

            # Store results in session
            filename_mapping = {item['stored_filename']: item['display_filename'] for item in current_batch_results}
            session['classification_results_batch'] = current_batch_results
            session['pdf_urls_current_batch'] = current_batch_pdfs
            session['filename_mapping_batch'] = filename_mapping

            return redirect(url_for('index'))
        # End of file processing loop (within POST)

    # --- GET Request Logic ---
    classification_results_display = session.get('classification_results_batch', [])
    pdf_urls_display = session.get('pdf_urls_current_batch', [])
    filename_mapping_display = session.get('filename_mapping_batch', {})

    # Clear session variables after retrieving
    if 'classification_results_batch' in session: session.pop('classification_results_batch')
    if 'pdf_urls_current_batch' in session: session.pop('pdf_urls_current_batch')
    if 'filename_mapping_batch' in session: session.pop('filename_mapping_batch')

    # --- DASHBOARD DATA ---
    total_resumes = 0
    role_counts = {}
    chart_labels_json = "[]" # Keep generating, even if not used by default
    chart_data_json = "[]"   # Keep generating, even if not used by default

    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file)
            if not df_log.empty and all(col in df_log.columns for col in LOG_COLUMNS):
                # Filter out invalid roles
                valid_roles_df = df_log[~df_log["predicted_role"].astype(str).str.lower().isin([
                    "error", "processing error", "extraction error", "prediction error",
                    "prediction error (attribute)", "prediction error (value)", "prediction error (general)",
                    "unreadable/empty", "no features found", "extraction failed/info",
                    "not found", "n/a"
                ])]
                total_resumes = len(valid_roles_df)
                role_counts = valid_roles_df["predicted_role"].value_counts().to_dict()

                # Generate chart data JSON (kept for potential future use)
                if role_counts:
                    chart_labels_list = list(role_counts.keys())
                    chart_data_list = list(role_counts.values())
                    try:
                        chart_labels_json = json.dumps(chart_labels_list)
                        chart_data_json = json.dumps(chart_data_list)
                    except TypeError as json_err:
                        logging.error(f"Error converting chart data to JSON: {json_err}")
                        chart_labels_json = "[]"; chart_data_json = "[]"

            elif not df_log.empty:
                 logging.warning(f"Dashboard: Log file {log_file} columns mismatch.")
        except pd.errors.EmptyDataError:
            logging.info("Dashboard: Log file is empty.")
        except Exception as e:
            logging.error(f"Error reading log file '{log_file}' for dashboard: {e}", exc_info=True)
            flash("Error updating dashboard data from log file.", "warning")
    else:
         logging.info("Dashboard: Log file not found.")

    # --- RENDER TEMPLATE ---
    classifier_ready = all([model_loaded, vectorizer_loaded, encoder_loaded])
    comparator_ready = vectorizer_loaded
    nltk_ok = nltk_data_verified and preprocessing_ready

    return render_template(
        "index.html",
        # Current batch data
        classification_results=classification_results_display,
        pdf_urls_current=pdf_urls_display,
        filename_mapping=filename_mapping_display,

        # Overall Dashboard Data
        total_resumes=total_resumes,
        role_counts=role_counts,
        chart_labels=chart_labels_json, # Pass even if not displayed by default
        chart_data=chart_data_json,     # Pass even if not displayed by default
        accuracy=accuracy,

        # Status flags
        classifier_ready=classifier_ready,
        comparator_ready=comparator_ready,
        nltk_ok=nltk_ok,
        model_name="Random Forest (TF-IDF)"
    )


# Route to handle the comparison form submission and show results
@app.route("/compare_results", methods=["POST"])
def compare_resumes():
    """Handles the comparison logic and renders results page."""
    # --- 1. Check Prerequisites ---
    if not vectorizer_loaded:
        flash("TF-IDF Vectorizer is not loaded. Cannot perform comparison.", "danger")
        return redirect(url_for('index') + '#comparator-section')
    if not preprocessing_ready:
        flash("NLTK preprocessing components failed to load. Comparison results may be less accurate.", "warning")

    # --- 2. Input Validation ---
    resume_file1 = request.files.get('resume1_compare')
    resume_file2 = request.files.get('resume2_compare')
    job_role_text = request.form.get('job_role_compare', '').strip()

    def flash_redirect(message, level="warning"):
        flash(message, level)
        return redirect(url_for('index') + '#comparator-section')

    if not resume_file1 or resume_file1.filename == '': return flash_redirect("Resume 1 is required.")
    if not allowed_file(resume_file1.filename): return flash_redirect(f"Invalid file type for Resume 1.")
    if not resume_file2 or resume_file2.filename == '': return flash_redirect("Resume 2 is required.")
    if not allowed_file(resume_file2.filename): return flash_redirect(f"Invalid file type for Resume 2.")
    if not job_role_text: return flash_redirect("Target Job Role / Description text is required.")

    # --- 3. File Handling (Temporary Save) ---
    filename1_orig = resume_file1.filename; filename2_orig = resume_file2.filename
    filename1_secure = secure_filename(filename1_orig); filename2_secure = secure_filename(filename2_orig)
    if not filename1_secure or not filename2_secure: return flash_redirect("Invalid characters detected in filename(s).", "danger")

    unique_id = uuid.uuid4().hex[:8]
    temp_filename1 = f"{unique_id}_{filename1_secure}"; temp_filename2 = f"{unique_id}_{filename2_secure}"
    path1 = os.path.join(app.config['COMPARE_FOLDER'], temp_filename1); path2 = os.path.join(app.config['COMPARE_FOLDER'], temp_filename2)
    temp_files_saved = False
    try:
        resume_file1.save(path1); resume_file2.save(path2); temp_files_saved = True
        logging.info(f"Saved comparison temp files: '{temp_filename1}', '{temp_filename2}'")
    except Exception as e:
        logging.error(f"Error saving comparison temp files: {e}", exc_info=True)
        # --- CORRECTED Cleanup Block 1 ---
        try:
            if os.path.exists(path1): os.remove(path1)
            if os.path.exists(path2): os.remove(path2)
        except Exception as cleanup_err:
             logging.warning(f"Could not clean up temp files after save error: {cleanup_err}")
        # --- End Correction ---
        return flash_redirect(f"Error saving temporary files: {e}", "danger")

    # --- 4. Processing: Text Extraction and Cleaning ---
    raw_text1 = extract_text_from_file(path1, temp_filename1); raw_text2 = extract_text_from_file(path2, temp_filename2)
    cleanup_needed = True # Flag to control temp file deletion

    # Check for extraction errors
    extract_errors = []
    if isinstance(raw_text1, str) and raw_text1.startswith("Error:"): extract_errors.append(f"Resume 1 ('{filename1_orig}'): {raw_text1}")
    if isinstance(raw_text2, str) and raw_text2.startswith("Error:"): extract_errors.append(f"Resume 2 ('{filename2_orig}'): {raw_text2}")
    if extract_errors:
        cleanup_needed = True # Still attempt cleanup
        if temp_files_saved:
             try:
                 # --- CORRECTED Cleanup Block 2 ---
                 if os.path.exists(path1): os.remove(path1)
                 if os.path.exists(path2): os.remove(path2)
                 # --- End Correction ---
                 logging.info("Cleaned up temp files after extraction error.")
             except Exception as e: logging.warning(f"Could not clean up temp files after extraction error: {e}")
        # Render the results page with an error message
        error_results = {"error": f"Text extraction failed: {'; '.join(extract_errors)}"}
        return render_template("results_compare.html", results=error_results)

    # Clean text using the same function as classifier, ensuring TF-IDF cleaning
    cleaned_resume1 = clean_text(raw_text1, for_tfidf=True); cleaned_resume2 = clean_text(raw_text2, for_tfidf=True)
    cleaned_job_role = clean_text(job_role_text, for_tfidf=True)

    # Check if cleaning resulted in empty strings
    empty_inputs = []
    if not cleaned_resume1 or not cleaned_resume1.strip(): empty_inputs.append(f"Resume 1 ('{filename1_orig}')")
    if not cleaned_resume2 or not cleaned_resume2.strip(): empty_inputs.append(f"Resume 2 ('{filename2_orig}')")
    if not cleaned_job_role or not cleaned_job_role.strip(): empty_inputs.append("Job Role Description")
    if empty_inputs:
        cleanup_needed = True # Still attempt cleanup
        if temp_files_saved:
             try:
                 # --- CORRECTED Cleanup Block 3 ---
                 if os.path.exists(path1): os.remove(path1)
                 if os.path.exists(path2): os.remove(path2)
                 # --- End Correction ---
                 logging.info("Cleaned up temp files after cleaning resulted in empty inputs.")
             except Exception as e: logging.warning(f"Could not clean up temp files after cleaning error: {e}")
        # Render results page with error
        error_results = {"error": f"Inputs became empty after cleaning: {', '.join(empty_inputs)}. Check content/stopwords."}
        return render_template("results_compare.html", results=error_results)

    # --- 5. Vectorization and Similarity ---
    similarity1, similarity2 = 0.0, 0.0; keywords1, keywords2 = [], []; kw_count1, kw_count2 = 0, 0
    try:
        vectors = vectorizer.transform([cleaned_resume1, cleaned_resume2, cleaned_job_role])
        vector1, vector2, vector_job_role = vectors[0], vectors[1], vectors[2]
        vec1_ok = hasattr(vector1, 'nnz') and vector1.nnz > 0; vec2_ok = hasattr(vector2, 'nnz') and vector2.nnz > 0
        job_vec_ok = hasattr(vector_job_role, 'nnz') and vector_job_role.nnz > 0

        if not job_vec_ok:
            logging.warning(f"Job role description resulted in zero vector: '{job_role_text[:100]}...'")
            flash("Job Role Description yielded no valid keywords after cleaning. Cannot compare.", "warning")
            # Proceed but similarities will be 0, keywords empty
        else:
            if vec1_ok: similarity1 = cosine_similarity(vector_job_role, vector1)[0][0]
            else: logging.warning(f"Resume 1 vector ('{filename1_orig}') is empty.")
            if vec2_ok: similarity2 = cosine_similarity(vector_job_role, vector2)[0][0]
            else: logging.warning(f"Resume 2 vector ('{filename2_orig}') is empty.")

            # Extract keywords only if vectors are valid
            if vec1_ok: keywords1, kw_count1 = get_matching_keywords(cleaned_resume1, cleaned_job_role, vector1, vectorizer)
            if vec2_ok: keywords2, kw_count2 = get_matching_keywords(cleaned_resume2, cleaned_job_role, vector2, vectorizer)

    except Exception as e:
        logging.error(f"Error during comparison vectorization/similarity: {e}", exc_info=True)
        cleanup_needed = True # Ensure cleanup happens
        if temp_files_saved:
             try:
                 # --- CORRECTED Cleanup Block 4 ---
                 if os.path.exists(path1): os.remove(path1)
                 if os.path.exists(path2): os.remove(path2)
                 # --- End Correction ---
                 logging.info("Cleaned up temp files after vectorization error.")
             except Exception as clean_e: logging.warning(f"Could not clean up temp files after vectorization error: {clean_e}")
        # Render results page with error
        error_results = {"error": f"Processing Error during comparison: {e}"}
        return render_template("results_compare.html", results=error_results)

    # --- 6. Prepare Output ---
    tolerance = 1e-6
    if abs(similarity1 - similarity2) < tolerance:
        best_match_message = f"Both resumes have a similar match score ({similarity1*100:.1f}%)."
        winner = 0
    elif similarity1 > similarity2:
        best_match_message = f"Resume 1 ('{filename1_orig}') is a better match ({similarity1*100:.1f}%) vs Resume 2 ({similarity2*100:.1f}%)."
        winner = 1
    else:
        best_match_message = f"Resume 2 ('{filename2_orig}') is a better match ({similarity2*100:.1f}%) vs Resume 1 ({similarity1*100:.1f}%)."
        winner = 2

    comparison_results_data = {
        "job_role": job_role_text, "filename1": filename1_orig, "filename2": filename2_orig,
        "similarity1": round(similarity1 * 100, 2), "similarity2": round(similarity2 * 100, 2),
        "keywords1": keywords1, "kw_count1": kw_count1, "keywords2": keywords2, "kw_count2": kw_count2,
        "best_match_message": best_match_message, "winner": winner
    }

    # --- 7. Cleanup ---
    if cleanup_needed and temp_files_saved:
        try:
            # --- CORRECTED Cleanup Block 5 ---
            if os.path.exists(path1): os.remove(path1)
            if os.path.exists(path2): os.remove(path2)
            # --- End Correction ---
            logging.info(f"Cleaned up comparison temp files.")
        except Exception as e:
            logging.warning(f"Could not clean up comparison temp files: {e}")

    # --- 8. Render Results ---
    return render_template("results_compare.html", results=comparison_results_data)


@app.route("/clear_dashboard", methods=["POST"])
def clear_dashboard():
    """Clears the classification log file and optionally uploaded files."""
    clear_files_too = request.form.get("clear_files") == "yes"
    log_cleared = False; files_deleted = 0; files_failed = 0

    if os.path.exists(log_file):
        try:
            pd.DataFrame(columns=LOG_COLUMNS).to_csv(log_file, index=False)
            logging.info(f"Cleared log file: {log_file}")
            log_cleared = True

            if clear_files_too:
                logging.warning("Clearing files from uploads folder...")
                upload_folder = app.config['UPLOAD_FOLDER']
                log_basename = os.path.basename(log_file)
                for filename in os.listdir(upload_folder):
                    if filename == log_basename: continue
                    file_path = os.path.join(upload_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path); files_deleted += 1
                    except Exception as e:
                        logging.error(f"Failed to delete {file_path}: {e}"); files_failed += 1
                logging.warning(f"Cleared {files_deleted} files. Failed: {files_failed}.")

            if log_cleared:
                flash_message = "Classification history cleared."
                if clear_files_too: flash_message += f" {files_deleted} file(s) deleted."
                if files_failed > 0: flash_message += f" ({files_failed} failed to delete)."
                flash(flash_message, "success")
            elif clear_files_too:
                 flash(f"Log not found. Cleared {files_deleted} files. Failed: {files_failed}.", "warning")

        except Exception as e:
            logging.error(f"Error clearing dashboard: {e}", exc_info=True)
            flash("Error clearing history.", "danger")
    else:
         flash("Log file not found.", "info")
         if clear_files_too:
             # Attempt file clearing even if log doesn't exist
             logging.warning("Log not found, attempting file clearing as requested...")
             upload_folder = app.config['UPLOAD_FOLDER']
             # Reset counters for this specific scenario
             files_deleted = 0; files_failed = 0
             for filename in os.listdir(upload_folder):
                 # No log file to skip this time
                 file_path = os.path.join(upload_folder, filename)
                 try:
                     if os.path.isfile(file_path) or os.path.islink(file_path):
                         os.unlink(file_path); files_deleted += 1
                 except Exception as e:
                     logging.error(f"Failed to delete {file_path}: {e}"); files_failed += 1
             logging.warning(f"Cleared {files_deleted} files. Failed: {files_failed}.")
             flash(f"Log not found. Cleared {files_deleted} files. {files_failed} failed.", "warning")

    return redirect(url_for("index"))


@app.route("/download/<category>")
def download_category(category):
    """Downloads resumes logged under a specific predicted category as a ZIP file."""
    if not os.path.exists(log_file):
        flash("Log file not found.", "warning"); return redirect(url_for('index'))
    try:
        df_log = pd.read_csv(log_file)
        if df_log.empty or not all(col in df_log.columns for col in LOG_COLUMNS):
             flash(f"No valid history found.", "info"); return redirect(url_for('index'))
    except pd.errors.EmptyDataError:
        flash(f"History log is empty.", "info"); return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error reading log file for download: {e}", exc_info=True)
        flash("Error reading log.", "danger"); return redirect(url_for('index'))

    matched_files_df = df_log[df_log["predicted_role"].astype(str) == str(category)]
    if matched_files_df.empty:
        flash(f"No resumes found for category '{category}'.", "info"); return redirect(url_for('index'))

    zip_buffer = io.BytesIO(); files_added = 0; files_missing = 0
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zip_file:
            for index, row in matched_files_df.iterrows():
                stored_fname = row["filename"]
                secure_fname = secure_filename(stored_fname)
                if not secure_fname or secure_fname != stored_fname:
                    logging.warning(f"Skipping invalid log filename: '{stored_fname}'")
                    files_missing += 1; continue

                fpath = os.path.join(app.config['UPLOAD_FOLDER'], secure_fname)
                if os.path.exists(fpath) and os.path.isfile(fpath):
                    arcname = secure_fname # Default to stored name
                    try: # Attempt reconstruction
                        name_part, ext_part = os.path.splitext(secure_fname)
                        if len(name_part) > 9 and name_part[-9] == '_' and len(name_part[-8:]) == 8:
                            int(name_part[-8:], 16) # Check hex
                            arcname = name_part[:-9] + ext_part
                    except (ValueError, IndexError): pass # Use default on error

                    zip_file.write(fpath, arcname=arcname); files_added += 1
                else:
                    files_missing += 1; logging.warning(f"Logged file not found: {fpath}")

        if files_added == 0:
             flash(f"Files for '{category}' logged but not found in storage.", "warning"); return redirect(url_for('index'))
        if files_missing > 0:
             flash(f"Downloaded {files_added} files for '{category}'. Note: {files_missing} logged file(s) missing.", "warning")
        # else: No success flash needed here, the download starts

    except Exception as e:
        logging.error(f"Error creating zip for {category}: {e}", exc_info=True)
        flash("Error creating download file.", "danger"); return redirect(url_for('index'))

    zip_buffer.seek(0)
    safe_category_name = re.sub(r'[^\w\-]+', '_', category)
    download_name = f"{safe_category_name}_resumes_{uuid.uuid4().hex[:4]}.zip"
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=download_name)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Integrated Resume Processing App (Charts Removed) ---")
    print(f"Flask Secret Key Loaded: {'Yes' if app.config['SECRET_KEY'] != 'dev-secret-key-replace-this' else 'No (Using default - CHANGE FOR PRODUCTION!)'}")
    print(f"Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"Compare Temp Folder: {os.path.abspath(app.config['COMPARE_FOLDER'])}")
    print(f"Log File: {os.path.abspath(log_file)}")
    print(f"--- NLTK Data Verified: {nltk_data_verified} | Preprocessing Ready: {preprocessing_ready} ---")
    print(f"--- Classifier Ready: {all([model_loaded, vectorizer_loaded, encoder_loaded])} | Comparator Ready: {vectorizer_loaded} ---")
    print(f"Model Accuracy: {accuracy}")
    app.run(debug=True, host='127.0.0.1', port=5000)