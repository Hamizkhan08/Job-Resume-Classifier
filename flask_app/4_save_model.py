import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os # Added for checking directories
import logging # Added for better logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- End Logging Setup ---


# --- NLTK Resource Downloads ---
# Ensure necessary NLTK data is available
resources = ['punkt', 'stopwords', 'wordnet']
logging.info("Checking NLTK resources...")
for resource in resources:
    try:
        # Adjust path finding for different resource types
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}')
        elif resource == 'stopwords':
             nltk.data.find(f'corpora/{resource}.zip') # Check for zip file
        elif resource == 'wordnet':
             nltk.data.find(f'corpora/{resource}.zip') # Check for zip file
        else:
             nltk.data.find(f'corpora/{resource}') # Generic fallback
        logging.info(f"NLTK resource '{resource}' found.")
    except LookupError:
        logging.warning(f"NLTK resource '{resource}' not found. Attempting download...")
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"NLTK resource '{resource}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Error downloading NLTK resource '{resource}': {e}. Script might fail if resource is critical.")
logging.info("NLTK resources check complete.")
# --- End NLTK Downloads ---

# --- Configuration ---
INPUT_CSV = "UpdatedResume.csv" # Make sure this file exists and has 'Resume_str' and 'Category'
OUTPUT_MODEL = 'best_model.pkl'
OUTPUT_VECTORIZER = 'tfidf_vectorizer.pkl'
OUTPUT_ENCODER = 'label_encoder.pkl'
OUTPUT_ACCURACY = 'model_accuracy.txt'
OUTPUT_PREDICTIONS = 'predicted_resumes.csv' # Optional output for verification
ARTIFACTS_DIR = "." # Save artifacts in the current directory (can be changed)

# Ensure output directory exists (important if ARTIFACTS_DIR is changed)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
logging.info(f"Artifacts will be saved in: {os.path.abspath(ARTIFACTS_DIR)}")

# --- Data Loading ---
logging.info(f"Loading dataset from '{INPUT_CSV}'...")
try:
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded dataset with shape {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Input CSV file '{INPUT_CSV}' not found. Please ensure it's in the correct directory and contains resume data.")
    exit(1) # Exit with error code
except Exception as e:
    logging.error(f"Error loading CSV file '{INPUT_CSV}': {e}", exc_info=True)
    exit(1)

# --- Basic Data Cleaning & Validation ---
logging.info("Performing initial data cleaning and validation...")
required_columns = ['Resume_str', 'Category']
if not all(col in df.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    logging.error(f"Error: Input CSV must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing_cols)}")
    exit(1)

initial_rows = len(df)
# Ensure 'Resume_str' is string type before checking length/NA
df['Resume_str'] = df['Resume_str'].astype(str)
df.dropna(subset=required_columns, inplace=True) # Drop rows where essential columns are NA
# Keep resumes with substantial content (adjust threshold if needed)
df = df[df['Resume_str'].str.strip().str.len() > 50]
df['Category'] = df['Category'].astype(str).str.strip() # Ensure Category is string and stripped

rows_after_na_len_filter = len(df)
logging.info(f"Rows dropped due to NA or short 'Resume_str': {initial_rows - rows_after_na_len_filter}")

if df.empty:
    logging.error("Error: No valid data remaining after initial cleaning. Check CSV content (needs 'Resume_str' and 'Category' columns with sufficient data).")
    exit(1)

logging.info(f"Shape after initial cleaning: {df.shape}")
logging.info(f"Value counts for 'Category' column (Top 20):\n{df['Category'].value_counts().head(20)}")
if len(df['Category'].unique()) > 50:
     logging.warning(f"Found {len(df['Category'].unique())} unique categories. Consider consolidating rare categories for better model performance.")

# --- Text Preprocessing ---
logging.info("Setting up text preprocessing components...")
# Check if stopwords are loaded correctly
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    logging.error("Could not load NLTK stopwords. Ensure they were downloaded correctly. Exiting.")
    exit(1)
lemmatizer = nltk.stem.WordNetLemmatizer()

# Common phrases to remove (ensure consistency with app.py)
common_phrases = [
    "curriculum vitae", "resume", "objective", "declaration",
    "personal information", "contact", "address", "phone", "email",
    "date of birth", "dob", "nationality", "gender", "marital status" # Add more if needed
]
logging.info(f"Phrases to be removed during cleaning: {common_phrases}")

def clean_text_for_training(text):
    """Cleans text specifically for TF-IDF model training."""
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove common phrases using regex word boundaries for better accuracy
    for phrase in common_phrases:
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

    if not text: # Handle cases where text becomes empty after initial cleaning
        return ""

    tokens = nltk.tokenize.word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and len(word) > 2 # Filter stopwords and short words
    ]
    return " ".join(filtered_tokens)

logging.info("Applying text cleaning to 'Resume_str' column...")
df['Cleaned_Resume'] = df['Resume_str'].apply(clean_text_for_training)

# Drop rows where cleaning resulted in empty string
rows_before_empty_clean = len(df)
df = df[df['Cleaned_Resume'].str.strip().str.len() > 0]
rows_after_empty_clean = len(df)
logging.info(f"Rows dropped due to empty result after cleaning: {rows_before_empty_clean - rows_after_empty_clean}")

if df.empty:
    logging.error("Error: No data remaining after text cleaning process. Check cleaning function or input data quality.")
    exit(1)
logging.info(f"Shape after text cleaning: {df.shape}")

# --- Feature Engineering (TF-IDF) ---
logging.info("Vectorizing cleaned text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased features slightly
    ngram_range=(1, 2),  # Consider single words and bigrams (adjust if needed)
    min_df=5,            # Ignore terms appearing in less than 5 documents (more robust)
    max_df=0.85,         # Ignore terms appearing in more than 85% of documents
    stop_words=None,     # Stopwords already removed in clean_text_for_training
    sublinear_tf=True    # Apply sublinear TF scaling (often improves performance)
)
try:
    X = vectorizer.fit_transform(df['Cleaned_Resume'])
    logging.info(f"TF-IDF Matrix created with shape: {X.shape}")
    if X.shape[1] == 0:
        logging.error("Error: TF-IDF resulted in zero features. Check cleaning function and data. Is the text becoming empty?")
        exit(1)
except MemoryError:
     logging.error("MemoryError during TF-IDF. Try reducing max_features, increasing min_df, or using a machine with more RAM.")
     exit(1)
except Exception as e:
    logging.error(f"Error during TF-IDF vectorization: {e}", exc_info=True)
    exit(1)

# --- Label Encoding ---
logging.info("Encoding 'Category' labels...")
label_encoder = LabelEncoder()
try:
    y = label_encoder.fit_transform(df['Category'])
    num_classes = len(label_encoder.classes_)
    logging.info(f"Encoded {num_classes} classes.")
    if num_classes < 50: # Log all classes if manageable
        logging.info(f"Classes: {', '.join(label_encoder.classes_)}")
    else: # Log only the count if too many
        logging.info("Classes encoded (list omitted due to length).")
except Exception as e:
    logging.error(f"Error during label encoding: {e}", exc_info=True)
    exit(1)

# --- Model Training (Random Forest Classifier) ---
logging.info("Splitting data for training and testing...")
# Stratify ensures class distribution is similar in train and test sets
test_set_size = 0.20 # Increased test size slightly for better evaluation
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_size, stratify=y, random_state=42
    )
    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
except ValueError as ve:
    if "n_splits=1" in str(ve) or "test_size=" in str(ve):
         logging.error(f"ValueError during train/test split: {ve}. This often happens with very small datasets or classes with only one sample. Ensure you have enough data per category.")
    else:
         logging.error(f"Error during train-test split: {ve}", exc_info=True)
    exit(1)
except Exception as e:
    logging.error(f"Error during train-test split: {e}", exc_info=True)
    exit(1)

logging.info("Training Random Forest model...")
# Adjusted parameters slightly based on common practices
rf_model = RandomForestClassifier(
    n_estimators=250,       # Slightly fewer trees, can increase if needed
    max_depth=None,         # Allow trees to grow fully (or set a limit like 30)
    min_samples_split=5,    # Min samples to split an internal node
    min_samples_leaf=3,     # Increased slightly to prevent overfitting on noise
    class_weight='balanced',# Adjusts weights inversely proportional to class frequencies
    random_state=42,        # For reproducibility
    n_jobs=-1,              # Use all available CPU cores for parallelism
    oob_score=True          # Enable Out-of-Bag score for estimation without test set
)

try:
    rf_model.fit(X_train, y_train)
    logging.info("Model training complete.")
    if hasattr(rf_model, 'oob_score_'):
        logging.info(f"Out-of-Bag (OOB) Score Estimate: {rf_model.oob_score_:.4f}")
except Exception as e:
    logging.error(f"Error during model training: {e}", exc_info=True)
    exit(1)

# --- Evaluation ---
logging.info("Evaluating model performance on the test set...")
accuracy = 0.0 # Initialize accuracy
try:
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Use zero_division=0 or 1 to handle cases where a class might have no predictions/support in the test set
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, digits=3)

    logging.info(f"\n--- Evaluation Results ---")
    logging.info(f"✅ Random Forest Test Accuracy: {accuracy:.4f}")
    logging.info(f"\n📊 Classification Report (Test Set):\n{report}")
    logging.info(f"--------------------------")

except Exception as e:
    logging.error(f"Error during model evaluation: {e}", exc_info=True)
    # Continue to saving artifacts even if evaluation fails, but log the error

# --- Saving Artifacts ---
logging.info("Saving model artifacts...")
try:
    # Ensure paths are joined correctly using os.path.join
    model_path = os.path.join(ARTIFACTS_DIR, OUTPUT_MODEL)
    vectorizer_path = os.path.join(ARTIFACTS_DIR, OUTPUT_VECTORIZER)
    encoder_path = os.path.join(ARTIFACTS_DIR, OUTPUT_ENCODER)
    accuracy_path = os.path.join(ARTIFACTS_DIR, OUTPUT_ACCURACY)

    with open(model_path, 'wb') as model_file: pickle.dump(rf_model, model_file)
    with open(vectorizer_path, 'wb') as vec_file: pickle.dump(vectorizer, vec_file)
    with open(encoder_path, 'wb') as enc_file: pickle.dump(label_encoder, enc_file)
    # Save only the accuracy score number for easy parsing in app.py
    with open(accuracy_path, "w") as f: f.write(f"{accuracy:.4f}")

    logging.info(f"✅ Successfully saved: {OUTPUT_MODEL}, {OUTPUT_VECTORIZER}, {OUTPUT_ENCODER}, {OUTPUT_ACCURACY} to {os.path.abspath(ARTIFACTS_DIR)}")
except Exception as e:
    logging.error(f"Error saving artifacts: {e}", exc_info=True)

# --- Optional: Save Full Predictions (for verification) ---
if OUTPUT_PREDICTIONS:
    logging.info("Predicting categories for the full dataset (optional)...")
    try:
        # Use the already vectorized 'X' for prediction on the whole dataset
        full_predictions_encoded = rf_model.predict(X)
        # Ensure prediction length matches dataframe length
        if len(full_predictions_encoded) == len(df):
            df['Predicted_Category'] = label_encoder.inverse_transform(full_predictions_encoded)
            # Select relevant columns to save
            output_df = df[['Resume_str', 'Category', 'Predicted_Category']].copy()
            predictions_path = os.path.join(ARTIFACTS_DIR, OUTPUT_PREDICTIONS)
            output_df.to_csv(predictions_path, index=False)
            logging.info(f"📁 Full predictions saved to '{predictions_path}'")
        else:
            logging.warning(f"Mismatch in prediction length ({len(full_predictions_encoded)}) and dataframe length ({len(df)}). Skipping saving full predictions.")
    except Exception as e:
        logging.error(f"Error saving full predictions: {e}", exc_info=True)

logging.info("\n--- Model training script finished ---")