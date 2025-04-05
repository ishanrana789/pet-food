a# Pet Food Recommender - ML Project Documentation (TXT Version - Revised)

## 1. Project Overview

This project implements a simple pet food recommender system using a K-Nearest Neighbors (KNN) machine learning algorithm. It takes a user's pet profile (species, life stage, size, allergies, special needs) and suggests suitable foods from a predefined dataset stored in a plain text file (`pet_food_data.txt`). The data within the text file uses a Comma Separated Value (CSV) structure. The project also includes a simulation of fetching additional nutritional data via a mock API call. This version includes fixes for numerical type errors during KNN processing and enhanced robustness.

## 2. Creating the Synthetic Dataset (`pet_food_data.txt`)

Since a real-world, readily usable pet food dataset is often hard to find, we created a synthetic one for this project, stored in a simple text file format for broader compatibility.

**Data Format:**
The file `pet_food_data.txt` contains data structured as **Comma Separated Values (CSV)**.
*   The first line contains the headers (column names), separated by commas.
*   Each subsequent line represents one pet food product, with its attribute values separated by commas.
*   Values containing commas themselves (like "Small,Medium") should ideally be enclosed in double quotes (standard CSV practice), which `pandas.read_csv` handles.

**Conceptualization Process:**

1.  **Identify Key Features:** Species, Life Stage, Size, Key Ingredient, Special Needs, Allergen Info, FoodID, FoodName, MockAPI_Protein, MockAPI_Fat.
2.  **Determine Categories/Values:** Listed common values for each feature (Dog/Cat, Puppy/Kitten/Adult/Senior, Chicken/Beef/Fish, Grain-Free/Contains Grain etc.).
3.  **Populate Data:** Created rows representing fictional products with variety.
4.  **Format as CSV in TXT:** Structured the data with commas as delimiters, suitable for saving in a `.txt` file and parsing with standard CSV readers.

**Self-Creation Guidelines:**
*   Use a plain text editor (Notepad, TextEdit, VS Code, etc.).
*   Make the first line your headers, separated by commas.
*   Each following line is a data record; separate values with commas.
*   If a value naturally contains a comma (e.g., "Weight Management, Joint Support"), enclose that entire value in double quotes: `"Weight Management, Joint Support"`.
*   Save the file with the `.txt` extension (e.g., `pet_food_data.txt`).
*   Ensure consistent structure and number of columns per row.

## 3. Machine Learning Algorithm: K-Nearest Neighbors (KNN)

**Why KNN?**
KNN is a simple, instance-based learning algorithm suitable for this recommendation task. It recommends items (foods) that are most similar to the query (pet profile) based on their features.

**How it Works Conceptually:**
1.  **Feature Space:** Each pet food is represented as a point in a multi-dimensional space based on its features (Life Stage, Size, Special Needs).
2.  **Query Point:** The user's pet profile is also represented as a point in this space.
3.  **Find Neighbors:** KNN finds the 'K' food points closest to the pet's point using a distance metric (Cosine similarity in this case).
4.  **Recommendation:** These 'K' closest foods are the recommendations.

**Implementation Details (Using `.txt` data - Revised):**

1.  **Data Loading:** The `load_data` function uses `pandas.read_csv('pet_food_data.txt')`. Added `str()` conversions during list processing for robustness against unexpected data types.
2.  **Input Correction:** The `get_pet_profile` function now includes logic to automatically correct obvious mismatches between species and life stage (e.g., Dog/Kitten becomes Dog/Puppy) and warns the user.
3.  **Hard Constraints (Pre-filtering):** Filtering by Species and Allergens happens on the DataFrame *after* loading, using the `filter_foods` function. Added more specific print messages if foods are filtered out.
4.  **Feature Selection & Preprocessing:** The `preprocess_for_knn` function selects `LifeStage`, `Size`, `SpecialNeeds_list` and converts them into binary numerical features (e.g., `LifeStage_Puppy=1`, `Size_Small=1`). Operates on the filtered DataFrame. Handles potential non-list data in list columns.
5.  **Pet Profile Vector & KNN Execution:**
    *   The `recommend_food_knn` function creates a numerical vector for the pet profile based on its features.
    *   **Crucially**, both the food feature matrix (`food_features`) and the pet feature vector (`pet_feature_vector`) are explicitly converted to `dtype=float` using `.astype(float)` or `dtype=float` during creation. This resolves potential `ValueError: setting an array element with a sequence` errors by ensuring scikit-learn's KNN receives purely numerical input.
    *   A `NearestNeighbors` model (using `cosine` similarity) is fitted on the numerical food features.
    *   The `kneighbors` method finds the nearest food vectors to the pet vector.
    *   Added `try...except` blocks around the KNN `fit` and `kneighbors` calls to catch potential runtime errors during the ML step.
    *   Improved feedback messages provide more clarity if no foods are available at different stages (filtering, processing, KNN).
6.  **Output:** Recommended food names/IDs are retrieved based on the KNN results from the original DataFrame.

## 4. Code Structure (`recommender_ml_txt.py`)

*   **`load_data(filepath)`:** Loads data using `pd.read_csv`. Includes `fillna` and `astype(str)` for initial cleaning, and safer list processing using `str()`.
*   **`get_pet_profile()`:** Prompts user for pet details. Includes validation and automatic correction for Species/LifeStage mismatches.
*   **`preprocess_for_knn(df)`:** Creates binary feature columns (LifeStage, Size, Needs) for KNN. Includes checks for empty dataframes and robust list handling.
*   **`filter_foods(df, pet_profile)`:** Filters DataFrame based on species and allergies. Includes checks for empty results and more informative print statements.
*   **`recommend_food_knn(pet_profile, food_df_full, n_recommendations)`:** Orchestrates filtering, preprocessing, and KNN execution. **Ensures feature matrices/vectors are float type.** Includes error handling for KNN steps and improved feedback.
*   **`mock_api_call(food_id, df)`:** Simulates API call using DataFrame lookup. Includes robust error handling for invalid IDs or non-numeric data.
*   **`display_recommendations(recommendations_list, food_df_full)`:** Formats and prints results. Includes error handling for missing food details.
*   **`if __name__ == "__main__":` block:** Sets `text_file = 'pet_food_data.txt'`, controls execution flow, and includes checks for empty results before displaying.

## 5. Usage Instructions

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   Required libraries installed: Run `pip install pandas scikit-learn` (Note: `openpyxl` is *not* needed).
2.  **Dataset:** Ensure the `pet_food_data.txt` file (created by copying the data provided earlier) is in the same directory as the Python script (`recommender_ml_txt.py`).
3.  **Run the Script:** Open a terminal or command prompt, navigate to the directory containing the files, and run:
    ```bash
    python recommender_ml_txt.py
    ```
4.  **Follow Prompts:** Enter your pet's details. The script will output recommendations based on KNN similarity after filtering.

## 6. Limitations and Future Work

*   **Simplicity of KNN:** Relies heavily on feature engineering and the distance metric. Doesn't capture complex relationships well.
*   **Small/Synthetic Dataset:** Recommendations are limited by the data provided. Real-world data would be needed for practical use.
*   **Basic Feature Engineering:** Binary encoding is simple. Embeddings or TF-IDF (on ingredients) could be more powerful with richer data.
*   **Mock API:** API call is simulated.
*   **Limited Scope:** Doesn't consider price, brand, reviews, wet/dry, detailed ingredients, etc.

**Future Improvements:**
*   Integrate real food database/API.
*   Use more advanced ML (content-based, collaborative filtering, hybrid).
*   Incorporate more features.
*   Develop a user interface.
*   Allow user weighting of factors.
