import unittest

from gradescope_utils.autograder_utils.decorators import weight, number, partial_credit

import os
import pandas as pd
import numpy as np
import requests
import re
import random

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans


from cell_utils import (
        register_local_file, 
        extract_variables, 
        extract_initial_variables, 
        extract_cell_content_and_outputs,
        find_cells_with_text, 
        find_cells_by_indices,
        has_string_in_cell,
        has_string_in_code_cells,
        search_plots_in_extracted_vars,
        search_text_in_extracted_content,
        print_text_and_output_cells,
        print_code_and_output_cells)


class GradeAssignment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GradeAssignment, self).__init__(*args, **kwargs)
        self.notebook_path = None

        self.local_files = [
            "pokemon.csv",
        ]
        for file in self.local_files:
            register_local_file(file)

    @partial_credit(0.0)
    @number("6.5")
    def test_6_5(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "##### ***6.5 Compare the clustering results")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "### Extra Credit ✨ (7 points total)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']
        print("This problem will be graded manually. This is just for answer printing")
        
        print("[Relevant Text Cells]")
        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @partial_credit(1.0)
    @number("1.2")
    def test_df(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "1.2 Understanding the Dataset (1 pt)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "1.3 Explore the Data (3 pts)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)


        df_from_code = cell_vars.get("pokemon_data", None)

        if os.path.exists("pokemon.csv"):
            df_from_csv = pd.read_csv("pokemon.csv")
        else:
            df_from_csv = None
        col_list= ["pokemon_id", "pokemon_name", "base_attack", "base_defense", "base_stamina", "type", "rarity", "charged_moves", "fast_moves", "candy_required",
                   "distance", "max_cp", "attack_probability", "base_capture_rate", "base_flee_rate", "dodge_probability", "max_pokemon_action_frequency",
                   "min_pokemon_action_frequency", "found_egg", "found_evolution", "found_wild", "found_research", "found_raid", "found_photobomb"] 

        # search for shape, 200, 43
        search_200, _ = search_text_in_extracted_content(cell_texts, "1007")
        search_43, _ = search_text_in_extracted_content(cell_texts, "24")

        found_shape_val = search_200 and search_43
        val_list = []
        for i in range(len(col_list)):
            search_, _ = search_text_in_extracted_content(cell_texts, col_list[i])
            val_list.append(search_)
        found_cols = all(val_list)
        
        print("Found df: ", isinstance(df_from_code, pd.DataFrame))
        print("Found shape values: ", found_shape_val)
        print("Found cols: ", found_cols)
        
        total_score = 0.0
        if isinstance(df_from_code, pd.DataFrame):
            total_score += 0.25
        if found_shape_val:
            total_score += 0.25
        if found_cols:
            total_score += 0.5
        set_score(total_score)

    @partial_credit(3.0)
    @number("1.3")
    def test_output(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "1.3 Explore the Data (3 pts)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "Part B (8 Pts Total)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        typ_list= ['Psychic', 'Fighting', 'Water', 'Grass', 'Electric', 'Fire', 'Dragon', 'Rock', 'Normal', 'Steel', 'Poison', 'Bug', 'Flying', 'Ice', 'Fairy', 'Ghost', 'Dark', 'Ground']
        rarity  = ['Standard','Legendary', 'Mythic', 'Ultra beast']


        # search for shape, 1007, 24
        search_1, _ = search_text_in_extracted_content(cell_texts, "1007")
        search_2, _ = search_text_in_extracted_content(cell_texts, "24")
        print("Found shape 0: ", search_1)
        print("Found shape 1: ", search_2)
        for _ in range(5):
            # Randomly pick an item from typ_list
            typ_item = random.choice(typ_list)
            search_type, _ = search_text_in_extracted_content(cell_texts, typ_item)
            # Check if both items exist in the paragraph
            if search_type:
                test_1 = True
            else:
                test_1 = False
            print("typ_item is correct? ", test_1)

        for _ in range(2):
            # Randomly pick an item from rarity
            rarity_item = random.choice(rarity)
            search_ra, _ = search_text_in_extracted_content(cell_texts, rarity_item)
            # Check if both items exist in the paragraph
            if search_ra:
                test_2 = True
            else:
                test_2 = False
        print("rarity_item is correct? ", test_2)

        total_score = 0.0
        if search_1:
            total_score +=0.5
        if search_2:
            total_score +=0.5
        if test_1:
            total_score +=1
        if test_2:
            total_score +=1

        set_score(total_score)

    @partial_credit(3.0)
    @number("2.1")
    def test_preprocess(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "2.1 Preprocessing (3 pts)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "2.2 Splitting, Training, and Testing (6 pts)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        missing_data_gt = {'pokemon_id': 0, 'pokemon_name': 0, 'base_attack': 0, 'base_defense': 0, 'base_stamina': 0, 'type': 0, 'rarity': 0, 'charged_moves': 0, 'fast_moves': 0, 'candy_required': 536, 'distance': 0, 'max_cp': 0, 'attack_probability': 103, 'base_capture_rate': 103, 'base_flee_rate': 103, 'dodge_probability': 103, 'max_pokemon_action_frequency': 103, 'min_pokemon_action_frequency': 103, 'found_egg': 263, 'found_evolution': 263, 'found_wild': 263, 'found_research': 263, 'found_raid': 263, 'found_photobomb': 263}
        missing_data_cell = cell_vars.get("missing_data", None)
        pokemon_data_cell = cell_vars.get("pokemon_data", None)
        exists_df = (missing_data_cell is not None) and isinstance(missing_data_cell, dict)
        
        missing_check = True if missing_data_gt == missing_data_cell else False


        nan_check = pokemon_data_cell.isna().any().any()  # or df.isnull()
        
        total_score = 0
        if exists_df:
            total_score+=1
        print("Found missing_data: ", exists_df)

        if ~nan_check:
            total_score+=1
        print("Found any null: ", nan_check)

        if missing_check:
            total_score+=1
        print("Found missing_data correct? ", missing_check)
        set_score(total_score)

    @partial_credit(6.0)
    @number("2.2")
    def test_split(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "2.2 Splitting, Training, and Testing (6 pts)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Part C (15 Pts Total)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        X_cell = cell_vars.get("X", None)
        y_cell = cell_vars.get("Y", cell_vars.get("y"))
        X_train_cell= cell_vars.get("X_train", None)
        X_test_cell= cell_vars.get("X_test", None)
        y_train_cell= cell_vars.get("y_train", None)
        y_test_cell= cell_vars.get("y_test", None)
        X_train_scaled_cell= cell_vars.get("X_train_scaled", None)
        X_test_scaled_cell= cell_vars.get("X_test_scaled", None)

        case_1 = X_cell.shape == (1007, 19)
        print("X shape and processed:", case_1)

        case_2 = False
        try:
            if 64 == (y_cell == True).sum():
                case_2 = True
        except ValueError:
            print("Oops!  That was no valid y_cell...")
        print("y shape and processed:", case_2)
        case_3 = X_train_cell.shape == (805, 19)
        print("X_train shape:", case_3)

        case_4 = X_test_cell.shape == (202, 19)
        print("X_test shape:", case_4)

        case_5 = y_train_cell.shape == (805, )
        case_6 = y_test_cell.shape == (202, )
        print("y_train shape:", case_4)
        print("y_test shape:", case_4)

        # Function to check if the datasets were rescaled
        def check_standard_scaler(X, y):
            # Check mean and std deviation of X
            mean_X = np.mean(X, axis=0)
            std_X = np.std(X, axis=0)

            # Check mean and std deviation of y (if y is one-dimensional)
            mean_y = np.mean(y)
            std_y = np.std(y)
            print(mean_X, std_X, mean_y, std_y)
            
            # Check if the mean is close to 0 and std is close to 1
            if np.allclose(mean_X, 0, atol=1e-1) and np.allclose(std_X, 1, atol=1e-1):
                X_scaled = True
            else:
                X_scaled = False

            if np.allclose(mean_y, 0, atol=1e-1) and np.allclose(std_y, 1, atol=1e-1):
                y_scaled = True
            else:
                y_scaled = False
            
            return X_scaled, y_scaled
        case_7, case_8 = False, False

        try:
            case_7, case_8 = check_standard_scaler(X_train_scaled_cell, X_test_scaled_cell)
        except ValueError:
            print("Oops!  That was no valid X_train_scaled_cell and X_test_scaled_cell...")
        print("X_train_scaled:", case_7)
        print("X_test_scaled:", case_8)

        total_score = 0.0
        if case_1:
            total_score += 1
        if case_2:
            total_score += 1

        if case_3 and case_4:
            total_score += 1
        if case_5 and case_6:
            total_score += 1

        if case_7:
            total_score += 1
        if case_8:
            total_score += 1
        set_score(total_score)


    @partial_credit(4.0)
    @number("3.1")
    def test_load_model(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***3.1 Load Models (4 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "***3.2 K-fold Cross-Validation (8 pts)***")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        models_cell = cell_vars.get("models", None)

        # Calculate flag: True if all required columns are present in new_df, False otherwise
        try:
            if any(isinstance(model, KNeighborsClassifier) for model in models_cell.values()):
                case_1 = True
        except ValueError:
            print("Oops!  That was no valid KNeighborsClassifier...")
        
        
        try:
            if any(isinstance(model, DecisionTreeClassifier) for model in models_cell.values()):
                case_2 = True
        except ValueError:
            print("Oops!  That was no valid DecisionTreeClassifier...")
        
        try:
            if any(isinstance(model, LogisticRegression) for model in models_cell.values()):
                case_3 = True
        except ValueError:
            print("Oops!  That was no valid LogisticRegression...")
        
        try:
            if any(isinstance(model, RandomForestClassifier) for model in models_cell.values()):
                case_4 = True
        except ValueError:
            print("Oops!  That was no valid LogisticRegression...")
        print("Found model 1:", case_1)
        print("Found model 2:", case_2)
        print("Found model 3:", case_3)
        print("Found model 4:", case_4)
        
        total_score = 0.0
        if case_1:
            total_score += 1
        if case_2:
            total_score += 1

        if case_3:
            total_score += 1
        if case_4:
            total_score += 1
        set_score(total_score)

    @partial_credit(6.0)
    @number("4.1")
    def test_model_report(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***4.1 Model Evaluation (6 pts)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "***4.2 Model Interpretation (6 pts)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        models_cell = cell_vars.get("models", None)
        
        total_score = 0.0
        # Check if models have been fitted
        for name, model in models_cell.items():
            try:
                # Attempt to check if the model is fitted
                check_is_fitted(model)
                print(f"{name} has been fitted.")
                total_score +=1
            except NotFittedError as exc:
                print(f"{name} has NOT been fitted.")

        search_99, _ = search_text_in_extracted_content(cell_texts, "0.99")
        search_98, _ = search_text_in_extracted_content(cell_texts, "0.98")
        search_97, _ = search_text_in_extracted_content(cell_texts, "0.97")
        search_96, _ = search_text_in_extracted_content(cell_texts, "0.96")
        search_95, _ = search_text_in_extracted_content(cell_texts, "0.96")

        if search_99 or search_98:
            total_score += 2
        elif search_97 or search_96:
            total_score += 1.5
        elif search_95:
            total_score += 1.0
        print("Model precision:", search_95, search_96, search_97, search_98, search_99)

        set_score(total_score)


    @partial_credit(10.0)
    @number("5.1")
    def test_feature(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***5.1 What are your top 4 features used to predict Pokemon Type (10 pts)?***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "Part F (17 Points Total)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        

        distance, _ = search_text_in_extracted_content(cell_texts, "distance")
        max_cp, _ = search_text_in_extracted_content(cell_texts, "max_cp")
        base_defense, _ = search_text_in_extracted_content(cell_texts, "base_defense")
        base_attack, _ = search_text_in_extracted_content(cell_texts, "base_attack")

        total_score = 0.0
        if distance:
            total_score += 2.5
        if max_cp:
            total_score += 2.5
        if base_defense:
            total_score += 2.5
        if base_attack:
            total_score += 2.5
        print("1st feature found:", distance)
        print("2nd feature found:", max_cp)
        print("3rd feature found:", base_defense)
        print("4th feature found:", base_attack)

        set_score(total_score)


    @partial_credit(1.0)
    @number("6.2.1")
    def test_extracted_df(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***6.2.1 Extract top features (1 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "***6.2.2 Normalize the data using StandardScaler. (1 pts)***")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        
        features = ['distance', 'max_cp', 'base_defense','base_attack']
        X_cell = cell_vars.get("X", None)


        total_score = 0.0
        # Check each feature individually and assign partial credit
        for feature in features:
            if feature in X_cell.columns:
                print(f"Feature is present.")
                total_score += 0.2
            else:
                print(f"This feature is NOT present.")

        # Additional check for a 5th feature
        if len(X_cell.columns) > 4:
            print("A 5th feature is present.")
        else:
            print("A 5th feature is NOT present. Good!")
            total_score += 0.2  # Adjust the score increment as needed

        set_score(total_score)

    @partial_credit(1.0)
    @number("6.2.2")
    def test_normalize_df(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "##### ***6.2.2 Normalize the data using StandardScaler. (1 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "There are varying reasons")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        X_scaled_cell = cell_vars.get("X_scaled", None)

        def check_standard_scaler(X):
            # Check mean and std deviation of X
            mean_X = np.mean(X, axis=0)
            std_X = np.std(X, axis=0)

            
            # Check if the mean is close to 0 and std is close to 1
            if np.allclose(mean_X, 0, atol=1e-1) and np.allclose(std_X, 1, atol=1e-1):
                X_scaled = True
            else:
                X_scaled = False
            
            return X_scaled
        
        total_score = 0.0
        try:
            case_1 = check_standard_scaler(X_scaled_cell)
            print("Congrat! Valid X_train_scaled_cell and X_test_scaled_cell...")
        except ValueError:
            print("Oops!  That was no valid X_train_scaled_cell and X_test_scaled_cell...")
        if case_1:
            total_score+=1
        set_score(total_score)

    @partial_credit(2.5)
    @number("6.3.1")
    def test_pca(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***6.3.1 Use a PCA function to return the transformed data. (2.5 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "***6.3.2 Extra Credit: Use a t-SNE function to return the transformed data. (2.5 pts)***")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        X_pca_cell = cell_vars.get("X_pca", None)
        pca_cell = cell_vars.get("pca", None)

        total_score = 0.0

        if X_pca_cell.shape == (1007, 2):
            total_score+=2.5/2
            print("X_pca shape correct")
        else:
            print("X_pca shape incorrect")
        if hasattr(pca_cell, 'explained_variance_ratio_'):
            total_score+=2.5/2
            print("pca used")
        else:
            print("pca not used")

        set_score(total_score)

    @partial_credit(2.5)
    @number("6.3.2")
    def test_tsne(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***6.3.2 Extra Credit: Use a t-SNE function to return the transformed data. (2.5 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "#### K-Means Clustering")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        X_tsne_cell = cell_vars.get("X_tsne", None)

        tsne_cell   = cell_vars.get("tsne", None)
        total_score = 0.0
        try:
            if X_tsne_cell.shape == (1007, 2):
                total_score+=2.5/2
                print("X_tsne shape correct")
        except ValueError:
            print("Oops!  That was no valid X_tsne...")
        
        try:
            if hasattr(tsne_cell, 'perplexity'):
                if hasattr(tsne_cell, 'perplexity'):
                    total_score+=2.5/2
                    print("tsne used")
        except ValueError:
            print("Oops!  tsne not used...")
        set_score(total_score)

    @partial_credit(3.0)
    @number("6.4")
    def test_kmeans(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "##### ***6.4 Apply K-means clustering and return the cluster labels (3 pts)***")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        print("check #1")

        end_cells = find_cells_with_text(self.notebook_path,  "Visualize your clustering!")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']
        print("check #2")

        # Extract variables and functions from the cells
        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        print("check #3")
        apply_kmeans_func = cell_vars.get("apply_kmeans", None)
        print("check #4")
        pca_labels = cell_vars.get("pca_labels", None)
        print("check #5")
        tsne_labels = cell_vars.get("tsne_labels", None)
        print("check #6")

        
        
        total_score = 0.0
        
        # Test the function implementation
        if apply_kmeans_func is not None:
            # Create test data
            X_test = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # Cluster 1
            [5, 5], [6, 5], [5, 6], [6, 6],  # Cluster 2
            [0, 5], [1, 5], [0, 6], [1, 6],  # Cluster 3
            [5, 0], [6, 0], [5, 1], [6, 1]   # Cluster 4
            ])
            n_clusters = 4
            seed = 42
            
            try:
                # Run student's function
                student_labels = apply_kmeans_func(X_test, n_clusters, seed)
                
                # Create our own KMeans with same parameters to compare
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
                expected_labels = kmeans.fit_predict(X_test)
                
                # Check if clustering is equivalent (allowing for label permutation)
                equivalent_clustering = True
                for i in range(len(X_test)):
                    for j in range(i + 1, len(X_test)):
                        if (student_labels[i] == student_labels[j]) != (expected_labels[i] == expected_labels[j]):
                            equivalent_clustering = False
                            break
                
                if equivalent_clustering:
                    total_score += 2.0
                    print("apply_kmeans function implementation correct")
                else:
                    print("apply_kmeans function produces incorrect clustering")
                    
            except Exception as e:
                print(f"Error in apply_kmeans function: {str(e)}")
        else:
            print("apply_kmeans function not found")
        
        # Test the actual outputs (pca_labels and tsne_labels)
        if isinstance(pca_labels, np.ndarray) and isinstance(tsne_labels, np.ndarray):
            if len(pca_labels) == len(tsne_labels) == 1007:  # Assuming this is the expected length
                total_score += 1.0
                print("Labels have correct length")
            else:
                print("Labels have incorrect length")
        else:
            print("Labels are not numpy arrays or are missing")
        
        set_score(total_score)


    @partial_credit(2.0)
    @number("7.2")
    def test_7_2(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "***7.2 Explore correlations")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path,  "**7.3 *Question*:")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']
        print("Cell Located Properly")

        # Extract variables and functions from the cells
        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        _, collection_list = search_plots_in_extracted_vars(cell_vars)

        figure_list = collection_list

        # Grading by outputs
        feature_list_gt = set({'distance', 'pokemon_id', 'rating', 'max_cp', 'base_stamina', 'base_attack', 'attack_probability', 'dodge_probability', 'base_defense'})


        score_from_output = 0
        score_from_figure = 0

        if True:
            try:
                contents = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
                # search each feature in the outputs
                corr_numbers = None
                feature_list = None
                max_test_score = 0

                corr_data_exists = None
                corr_data_in_range = None
                feature_list_correct = None
                for content in contents:
                    outputs = content["outputs"]

                    if outputs == None:
                        continue
                    for output in outputs:
                        if output == None:
                            continue 

                        numbers = re.findall(r'-?\d+\.\d+|-?\d+', output)
                        numbers_in_range = [float(num) for num in numbers if -1.01 < float(num) < 1.01]

                        cur_feature_list = set()
                        for feature in feature_list_gt:
                            if feature in output:
                                cur_feature_list.add(feature)


                        test_data_exists = len(numbers) > 0
                        test_data_count = len(numbers) >= 81
                        test_data_in_range = len(numbers_in_range) >= 81
                        test_data_count_exact = len(numbers_in_range) == 81
                        test_feature_correct = cur_feature_list == feature_list_gt

                        test_score = int(test_data_exists) + int(test_data_count) + int(test_data_in_range) + int(test_data_count_exact) + int(test_feature_correct)

                        if test_score > max_test_score:
                            max_test_score = test_score
                            corr_numbers = numbers_in_range
                            feature_list = cur_feature_list

                            corr_data_exists = test_data_count
                            corr_data_in_range = test_data_in_range
                            feature_list_correct = test_feature_correct
                                                                                                                

                print("Correlation numbers found: ", corr_numbers)
                print("Features found: ", feature_list)
                print("Data exists? ", corr_data_exists)
                print("Data in range? ", corr_data_in_range)
                print("Feature List Correct? ", feature_list_correct)
                print("Rubric (Data exists & data in range): ", corr_data_exists & corr_data_in_range)


                if corr_data_exists and corr_data_in_range:
                    score_from_output = 2.0
                if feature_list_correct:
                    score_from_output = 2.0
            except:
                pass

        if True: 
            try:
                if len(figure_list) == 0:
                    print("No plots found")
                else:

                    best_figure = None
                    data_in_unit_range = None
                    feature_list_correct = None

                    test_max_score = 0
                    for figure in figure_list:

                        x_labels = figure["x_labels"]
                        y_labels = figure["y_labels"]
                        matrix_data = np.array(figure["matrix_data"])

                        figure_title = figure["figure_title"]
                        axis_title = figure["axis_title"]

                        size_x_label = len(x_labels)
                        size_y_label = len(y_labels)
                        size_matrix_data = matrix_data.shape[0] * matrix_data.shape[1]

                        feature_num = size_x_label
                        data_num = size_matrix_data

                

                        feature_list = set(x_labels)

                        test_value_in_unit_range = (matrix_data.min() > -1.01) and (matrix_data.max() < 1.01)
                        test_size_consistent = (data_num == size_x_label * size_y_label)
                        test_feature_num_in_range = 7 < feature_num < 11
                        test_feature_list = feature_list == feature_list_gt

                        test_score = test_value_in_unit_range + test_size_consistent + test_feature_num_in_range + test_feature_list


                        if test_score > test_max_score:
                            test_max_score = test_score
                            best_figure = figure

                            feature_list_correct = test_feature_list
                            data_in_unit_range = test_value_in_unit_range
                        
                        
                    print("Best figure found: ", best_figure)
                    print("Figure/Axis Indices:", best_figure["fig_index"], best_figure["ax_index"])
                    print("Features:", best_figure["x_labels"])
                    print("Data:\n", best_figure["matrix_data"])
                    print("Rubric 1: Data in unit range? ", data_in_unit_range)
                    print("Rubric 2: Feature list correct? ", feature_list_correct)

                    if data_in_unit_range:
                        score_from_figure += 1.0

                    if feature_list_correct:
                        score_from_figure += 1.0
            except:
                pass

        print("Score from output: ", score_from_output)
        print("Score from figure: ", score_from_figure)
        print("Final Score (Max): ", max(score_from_output, score_from_figure))

        set_score(max(score_from_output, score_from_figure))
