
# Job Resume Classifier & Comparator

A web application built with Flask and Bootstrap 5 to automatically classify uploaded resumes into job roles and compare the relevance of two resumes against a specific job description.

job.png

## About The Project

This project provides a user-friendly web interface for HR professionals, recruiters, or individuals to streamline the initial resume screening process. It leverages machine learning models to:

1.  **Classify:** Automatically categorize resumes (PDF, DOCX, TXT) into predefined job roles based on their content.
2.  **Compare:** Evaluate the similarity and relevance of two candidate resumes against a provided job description or set of required skills.

The application features a dashboard to visualize classification statistics and allows downloading categorized resumes for offline processing.

---

## ✨ Key Features

*   **Resume Upload:** Supports PDF, DOCX, and TXT file formats via drag-and-drop or traditional file input.
*   **Automatic Role Classification:** Predicts the job role for uploaded resumes using a pre-trained model.
*   **Two-Resume Comparison:** Compares two resumes against a target job description using text similarity techniques (likely TF-IDF/Cosine Similarity based on the comparator logic).
*   **Interactive Dashboard:** Displays:
    *   Total resumes processed.
    *   Most frequent job role identified.
    *   Counts for each classified role.
    *   Option to download resumes grouped by predicted role (ZIP archive).
*   **Inline PDF Preview:** Displays uploaded PDF resumes directly within the interface.
*   **System Status:** Shows the readiness of classifier/comparator components and NLTK data.
*   **Theme Switching:** Supports both Light and Dark themes for user preference.
*   **History Management:** Option to clear the dashboard statistics and optionally delete uploaded files from the server.

---

## 🛠️ Built With

*   **Backend:**
    *   [Python](https://www.python.org/) (Specify your version, e.g., 3.8+)
    *   [Flask](https://flask.palletsprojects.com/)
    *   [Scikit-learn](https://scikit-learn.org/) (for ML model loading and prediction)
    *   [NLTK](https://www.nltk.org/) (for text processing)
    *   [python-docx](https://python-docx.readthedocs.io/) (for reading .docx files)
    *   [PyPDF2](https://pypi.org/project/PyPDF2/) or similar (for reading .pdf files)
*   **Frontend:**
    *   HTML5
    *   CSS3 (with Custom Properties for Theming)
    *   JavaScript (Vanilla JS for interactions)
    *   [Bootstrap 5](https://getbootstrap.com/)
    *   [Font Awesome](https://fontawesome.com/) (for icons)
*   **Data Handling:**
    *   Likely uses Flask session or simple file storage for history/uploads.

---

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

*   **Python:** Version 3.8 or higher recommended. Download from [python.org](https://www.python.org/downloads/).
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** To clone the repository.
*   **Pre-trained Model Files:** You need the `.pkl` or `.joblib` files for the classifier model and the vectorizer. See the [Model Files](#-model-files) section below.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPOSITORY_NAME].git
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd [YOUR_REPOSITORY_NAME]
    ```
3.  **Create and activate a virtual environment (Recommended):**
    *   On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
4.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(You need to create a `requirements.txt` file first! Use `pip freeze > requirements.txt` after installing all necessary packages locally.)*

5.  **Download NLTK data:** Run this command in your terminal (within the activated virtual environment):
    ```sh
    python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger omw-1.4
    ```
    *(Check your `app.py` to confirm if other NLTK datasets are needed)*

6.  **Place Model Files:** Copy your pre-trained model (`.pkl`/`.joblib`) and vectorizer (`.pkl`/`.joblib`) files into the root directory of the project (or update the paths in `app.py` if you place them elsewhere). See [Model Files](#-model-files).

---

## 🔧 Usage

1.  **Run the Flask application:**
    ```sh
    flask run
    ```
    *Alternatively, you might run it directly:*
    ```sh
    python app.py
    ```
2.  **Open your web browser** and navigate to: `http://127.0.0.1:5000` (or the address provided by Flask).

3.  **Use the application:**
    *   **Classify:** Go to the "Classify Resumes" section, drag & drop or select your resume files (PDF, DOCX, TXT), and click "Classify Uploaded Resumes". Results will appear below, and PDFs can be previewed.
    *   **Compare:** Go to the "Compare Resumes" section, upload two resumes, paste the target job description, and click "Compare Resumes". (Comparison results display logic needs to be implemented or clarified).
    *   **Dashboard:** View statistics and download classified resumes from the "Dashboard" section.
    *   **Theme:** Toggle between light and dark themes using the sun/moon icon in the sidebar footer.
    *   **Clear History:** Use the "Clear History" button in the sidebar footer (optionally checking the box to delete uploaded files).

---

## 💾 Model Files

This application requires pre-trained machine learning models to function correctly. You **must** place the following files in the project's root directory (or modify the loading paths in `app.py`):

1.  **Classifier Model:** `[YOUR_CLASSIFIER_MODEL_FILENAME].pkl` (or `.joblib`) - The trained model responsible for predicting job roles.
2.  **Vectorizer:** `[YOUR_VECTORIZER_FILENAME].pkl` (or `.joblib`) - The TF-IDF Vectorizer (or similar) fitted on the training data, required for both classification and comparison to transform text into numerical features.

*Ensure these filenames match exactly what is being loaded in your `app.py` file.*


## 📧 Contact

Hamiz Khan 


