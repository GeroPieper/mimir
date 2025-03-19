import os
import json
import pickle
import random
import difflib
import shutil
import subprocess
import unicodedata
import multiprocessing
import concurrent.futures
from itertools import combinations
import logging
import numpy as np

import re
import pandas as pd
import requests
import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.tree
import sklearn.feature_selection

from typing import List, Tuple

import time
import math
import concurrent.futures
from pandas import Series

import dataset
import auto_instance
import pdep
import hpo
import helpers
import ml_helpers
import correctors

root_logger = logging.getLogger()
# Check if there are no handlers attached to the root logger
if not root_logger.hasHandlers():
    # Configure logging with your debug logging settings
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # Modify existing logging configuration to include debug logging settings
    root_logger.setLevel(logging.DEBUG)
    # Update format for existing handlers
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))


class Cleaning:
    """
    Class that carries out the error correction process.
    """

    # TODO vor hochladen entfernen
    time_llm_master: float = 0
    llm_master_errors: int = 0
    llm_master_errors_corrected: int = 0
    time_llm_transformation: float = 0
    llm_transformation_columns: int = 0
    llm_transformation_errors: int = 0
    llm_transformation_errors_corrected: int = 0
    time_llm_correction: float = 0
    llm_correction_errors: int = 0
    llm_correction_errors_corrected: int = 0
    time_correction: float = 0

    def __init__(self,
                 labeling_budget: int,
                 classification_model: str = 'ABC',
                 clean_with_user_input: bool = True,
                 feature_generators: List[str] = ['auto_instance', 'fd', 'llm_correction', 'llm_master',
                                                  'llm_transformation'],
                 last_try_threshold: float = 1.0,
                 llm_temp: int = 0,
                 llm_top_k: int = 1,
                 llm_gpu_layers: int = 48,
                 llm_ctx_size: int = 4096,
                 vicinity_orders: List[int] = [1],
                 vicinity_feature_generator: str = 'naive',
                 auto_instance_cache_model: bool = False,
                 n_best_pdeps: int = 3,
                 training_time_limit: int = 30,
                 synth_tuples: int = 100,
                 synth_cleaning_threshold: float = 0.9,
                 test_synth_data_direction: str = 'user_data',
                 pdep_features: Tuple[str] = ('pr',),
                 gpdep_threshold: float = 0.3,
                 fd_feature: str = 'norm_gpdep',
                 dataset_analysis: bool = False,
                 llm_name_corrfm: str = 'Meta-Llama-3-8B-Instruct.Q4_0.gguf',
                 sampling_technique: str = 'baran'):
        """
        Parameters of the cleaning experiment.
        @param labeling_budget: How many tuples are labeled by the user. In the Baran publication, 20  labels are frequently used.
        @param classification_model: "ABC" for sklearn's AdaBoostClassifier with n_estimators=100, "CV" for
        cross-validation. Default is "ABC".
        @param clean_with_user_input: Take user input to clean data with. This will always improve cleaning performance,
        and is recommended to set to True as the default value. Handy to disable when debugging models.
        @param feature_generators: Six feature generators are available: 'auto_instance', 'fd', 'vicinity',
        'llm_master', 'llm_correction', 'llm_transformation'.  Pass them as strings in a list to make Mimir use them, e.g.
        ['vicinity', 'auto_instance'].
        @param last_try_threshold: a threshold for trying to find a transformation in the last iteration of llm_transformation.
        @param llm_temp: temperature for the llm used in 'llm_master', 'llm_correction', 'llm_transformation'.
        @param llm_top_k: top_k for the llm used in 'llm_master', 'llm_correction', 'llm_transformation'.
        @param llm_gpu_layers: Number of layers used by llm on the gpu
        @param llm_ctx_size: Context size for the llm
        @param vicinity_orders: The pdep approach enables the usage of higher-order dependencies to clean data. Each
        order used is passed as an integer, e.g. [1, 2]. Unary dependencies would be used by passing [1] for example.
        @param vicinity_feature_generator: How vicinity features are generated. Either 'pdep' or 'naive'. Baran uses
        'naive' by default.
        @param auto_instance_cache_model: Whether or not the AutoGluon model used in the auto_instance model is stored after
        training and used from cache. Otherwise, a model is retrained with every user-provided tuple. If true, reduces
        cleaning time significantly.
        @param n_best_pdeps: When using vicinity_feature_generator = 'pdep', dependencies are ranked via the pdep
        measure. After ranking, the n_best_pdeps dependencies are used to provide cleaning suggestions. A good heuristic
        is to set this to 3.
        @param training_time_limit: Limit in seconds of how long the AutoGluon imputer model is trained.
        @param synth_tuples: maximum number of tuples to infer training data from.
        @param synth_cleaning_threshold: Threshold for column-cleaning to pass in order to leverage synth-data.
        Deactivates if set to -1.
        @param test_synth_data_direction: Direction in which the synth data's usefulness for cleaning is being tested.
        Set either to 'user_data' to clean user-data with synth-data. Or 'synth_data' to clean synth-data with
        user-inputs.
        @param pdep_features: List of features the pdep-feature-generator will return. Can be
        'pr' for conditional probability, 'vote' for how many FDs suggest the correction, 'pdep' for the
        pdep-score of the dependency providing the correction, and 'gpdep' for the gpdep-socre of said
        dependency.
        @param gpdep_threshold: Threshold a suggestion's gpdep score must pass before it is used to generate a feature.
        @param fd_feature: Feature used by the the fd_instance imputer to make cleaning suggestions. Choose from ('gpdep', 'pdep', 'fd').
        @param dataset_analysis: Write a detailed analysis of how Mimir cleans a a dataset to a .json file.
        @param llm_name_corrfm: Name of the OpenAI LLM model used in et_corrfm.
        @param sampling_technique: Technique used to sample row for user input.
        """

        self.SYNTH_TUPLES = synth_tuples
        self.CLEAN_WITH_USER_INPUT = clean_with_user_input
        self.SAMPLING_TECHNIQUE = sampling_technique
        self.FEATURE_GENERATORS = feature_generators
        self.LAST_TRY_THRESHOLD = last_try_threshold
        self.LLM_TEMP = llm_temp
        self.LLM_TOP_K = llm_top_k
        self.LLM_GPU_LAYERS = llm_gpu_layers
        self.LLM_CTX_SIZE = llm_ctx_size
        self.VICINITY_ORDERS = vicinity_orders
        self.VICINITY_FEATURE_GENERATOR = vicinity_feature_generator
        self.AUTO_INSTANCE_CACHE_MODEL = auto_instance_cache_model
        self.N_BEST_PDEPS = n_best_pdeps
        self.TRAINING_TIME_LIMIT = training_time_limit
        self.CLASSIFICATION_MODEL = classification_model
        self.SYNTH_CLEANING_THRESHOLD = synth_cleaning_threshold
        self.TEST_SYNTH_DATA_DIRECTION = test_synth_data_direction
        self.PDEP_FEATURES = pdep_features
        self.GPDEP_THRESHOLD = gpdep_threshold
        self.FD_FEATURE = fd_feature
        self.DATASET_ANALYSIS = dataset_analysis
        self.MAX_VALUE_LENGTH = 50
        self.LABELING_BUDGET = labeling_budget
        self.LLM_NAME_CORRFM = llm_name_corrfm
        self.logger = logging.getLogger(__name__)

        # TODO remove unused attributes
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = False
        self.SAVE_RESULTS = False

        # variable for debugging CV
        self.sampled_tuples = 0

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        This method incrementally adds a key-value into a dictionary-implemented model.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    @staticmethod
    def _value_encoder(value, encoding):
        """
        This method represents a value with a specified value abstraction encoding method.
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    def _value_based_models_updater(self, models, ud):
        """
        This method updates the value-based error corrector models with a given update dictionary.
        """
        remover_transformation = {}
        adder_transformation = {}
        replacer_transformation = {}
        s = difflib.SequenceMatcher(None, ud["old_value"], ud["new_value"])
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            index_range = json.dumps([i1, i2])
            if tag == "delete":
                remover_transformation[index_range] = ""
            if tag == "insert":
                adder_transformation[index_range] = ud["new_value"][j1:j2]
            if tag == "replace":
                replacer_transformation[index_range] = ud["new_value"][j1:j2]
        for encoding in ["identity", "unicode"]:
            encoded_old_value = self._value_encoder(ud["old_value"], encoding)
            if remover_transformation:
                self._to_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation))
            if adder_transformation:
                self._to_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation))
            if replacer_transformation:
                self._to_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation))
            self._to_model_adder(models[3], encoded_old_value, ud["new_value"])

    def _value_based_corrector(self, models, ed):
        """
        This method takes the value-based models and an error dictionary to generate potential value-based corrections.
        """
        results = {'remover': [], 'adder': [], 'replacer': [], 'swapper': []}
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in ["identity", "unicode"]:
                results_dictionary = {}
                encoded_value_string = self._value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[
                                                                                      change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            pr = model[encoded_value_string][transformation_string] / sum_scores
                            results_dictionary[new_value] = pr
                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            pr = model[encoded_value_string][new_value] / sum_scores
                            results_dictionary[new_value] = pr
                results[model_name].append(results_dictionary)
        return results

    def initialize_dataset(self, d):
        """
        This method initializes the dataset.
        """
        d.results_folder = os.path.join(os.path.dirname(d.path), "mirmir-results-" + d.name)
        if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.corrected_cells = {} if not hasattr(d, "corrected_cells") else d.corrected_cells
        d.column_categories = {} if not hasattr(d, "column_categories") else d.column_categories
        return d

    def categorize_columns(self, d):
        """
        Categorizes each column in the dataframe and stores the result in d.column_categories.
        """

        self.logger.debug("Start categorizing columns.")
        d.column_categories = {}
        x = 0
        for column_name in d.dataframe.columns:
            column = d.dataframe[column_name]
            category = self.detect_column_category(column)
            d.column_categories[x] = column_name, category
            self.logger.debug(f"Spalte {column_name} hat Kategorie {category}")
            x += 1
        self.logger.debug("Finished categorizing columns.")

    def detect_column_category(self, column: Series) -> str:
        """
        Determines the category of a given column based on its data type.

        :param column: A pandas Series representing the column to classify.
        :return: One of "Numeric", "String", "DateTime", "Boolean", or "Unknown".
        """

        def looks_like_datetime(value):
            patterns = [
                # Date patterns
                r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # 1/1/04 or 01/01/2004
                r"^\d{4}-\d{2}-\d{2}$",  # 2023-01-01
                r"^\d{2}\.\d{2}\.\d{2,4}$",  # 01.01.23 or 01.01.2023
                r"^\d{1,2}-\d{1,2}-\d{2,4}$",  # 1-1-04 or 01-01-2023
                r"^\d{8}$",  # 01012023

                # Time patterns
                r"^\d{2}:\d{2}(:\d{2})?$",  # 15:30 or 15:30:00
                r"^\d{2}:\d{2}:\d{2}\.\d{3}$",  # 15:30:45.123 (with milliseconds)
                r"^\d{2}\d{2}$",  # 1530 (HHMM)

                # Date and Time combinations
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$",  # 2023-01-01T15:30
                r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$",  # 2023/01/01 15:30
                r"^\d{1,2}-\w{3}-\d{2,4}$",  # 1-Jan-2023 or 01-Jan-23

                # Month and Year
                r"^\w{3,9} \d{4}$",  # January 2023
                r"^\d{4} \w{3,9}$",  # 2023 January
            ]
            for pattern in patterns:
                if re.match(pattern, str(value)):
                    return True
            return False

        # Check if the column contains Boolean values
        if pd.api.types.is_bool_dtype(column):
            return "Boolean"
        # Attempt to convert strings to numeric and check again
        elif pd.api.types.is_numeric_dtype(column) or pd.to_numeric(column, errors="coerce").notna().all():
            return "Numeric"
        # Check for strings and potential datetimes
        elif pd.api.types.is_string_dtype(column):
            if column.apply(looks_like_datetime).mean() > 0.8:
                return "DateTime"
            return "String"
        # Check for existing datetime types
        elif pd.api.types.is_datetime64_any_dtype(column):
            return "DateTime"
        else:
            return "Unknown"

    def initialize_models(self, d):
        """
        This method initializes the error corrector models.
        """
        # Initialize LLM cache
        conn = helpers.connect_to_cache()
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS cache
                          (dataset TEXT,
                          row INT,
                          column INT,
                          correction_model TEXT,
                          correction_tokens TEXT,
                          token_logprobs TEXT,
                          top_logprobs TEXT)''')
        conn.commit()

        # Correction store for feature creation
        corrections_features = []  # don't need further processing being used in ensembling.

        for feature in self.FEATURE_GENERATORS:
            if feature == 'vicinity':
                if self.VICINITY_FEATURE_GENERATOR == 'pdep':
                    for o in self.VICINITY_ORDERS:
                        corrections_features.append(f'vicinity_{o}')
                elif self.VICINITY_FEATURE_GENERATOR == 'naive':  # every lhs combination creates a feature
                    _, cols = d.dataframe.shape
                    for o in self.VICINITY_ORDERS:
                        for lhs in combinations(range(cols), o):
                            corrections_features.append(f'vicinity_{o}_{str(lhs)}')
            elif feature == 'value':
                models = ['remover', 'adder', 'replacer', 'swapper']
                encodings = ['identity', 'unicode']
                models_names = [f"value_{m}_{e}" for m in models for e in encodings]
                corrections_features.extend(models_names)
            else:
                corrections_features.append(feature)

        d.corrections = helpers.Corrections(corrections_features)
        d.inferred_corrections = helpers.Corrections(corrections_features)

        d.vicinity_models = {}
        d.value_models = [{}, {}, {}, {}]

        d.lhs_values_frequencies = {}
        if 'vicinity' in self.FEATURE_GENERATORS:
            for o in self.VICINITY_ORDERS:
                d.vicinity_models[o], d.lhs_values_frequencies[o] = pdep.mine_all_counts(
                    df=d.dataframe,
                    detected_cells=d.detected_cells,
                    order=o,
                    ignore_sign=self.IGNORE_SIGN)
        d.imputer_models = {}

    def sample_tuple(self, d):
        """
        This method samples tuples to be corrected by the user. It's a simply greedy algorithm that selects
        tuples containing the highest number of errors for labeling.
        """
        self.logger.debug('Start tuple sampling.')
        sampled_tuples: List[int] = []  # list of sampled rows

        if self.SAMPLING_TECHNIQUE == 'greedy':
            error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)
            errors_by_row = {row: len(cells) for row, cells in error_positions.original_row_errors().items()}.items()
            random.shuffle(list(errors_by_row))  # remove order by index
            ordered_errors_by_row = sorted(errors_by_row, key=lambda x: x[1],
                                           reverse=True)  # establish order by error count

            sampled_tuples = [x[0] for x in ordered_errors_by_row[:self.LABELING_BUDGET]]

        elif self.SAMPLING_TECHNIQUE == 'baran':
            rng = np.random.default_rng(seed=random_seed)
            remaining_column_unlabeled_cells = {}
            remaining_column_unlabeled_cells_error_values = {}
            remaining_column_uncorrected_cells = {}
            remaining_column_uncorrected_cells_error_values = {}

            error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)

            column_errors = error_positions.original_column_errors()

            for j in column_errors:
                for error_cell in column_errors[j]:
                    if error_cell not in d.corrected_cells:  # no correction suggestion has been found yet.
                        self._to_model_adder(remaining_column_uncorrected_cells, j, error_cell)
                        self._to_model_adder(remaining_column_uncorrected_cells_error_values, j,
                                             d.dataframe.iloc[error_cell])
                    if error_cell not in d.labeled_cells:
                        # the cell has not been labeled by the user yet. this is stricter than the above condition.
                        self._to_model_adder(remaining_column_unlabeled_cells, j, error_cell)
                        self._to_model_adder(remaining_column_unlabeled_cells_error_values, j,
                                             d.dataframe.iloc[error_cell])
            tuple_score = np.ones(d.dataframe.shape[0])
            tuple_score[list(d.labeled_tuples.keys())] = 0.0

            if len(remaining_column_uncorrected_cells) > 0:
                remaining_columns_to_choose_from = remaining_column_uncorrected_cells
                remaining_cell_error_values = remaining_column_uncorrected_cells_error_values
            else:
                remaining_columns_to_choose_from = remaining_column_unlabeled_cells
                remaining_cell_error_values = remaining_column_unlabeled_cells_error_values

            for j in remaining_columns_to_choose_from:
                for cell in remaining_columns_to_choose_from[j]:
                    value = d.dataframe.iloc[cell]
                    column_score = math.exp(len(remaining_columns_to_choose_from[j]) / len(column_errors[j]))
                    cell_score = math.exp(
                        remaining_cell_error_values[j][value] / len(remaining_columns_to_choose_from[j]))
                    tuple_score[cell[0]] *= column_score * cell_score

            for _ in range(self.LABELING_BUDGET):  # sample 20 different tuples
                max_args = np.argwhere(tuple_score == np.amax(tuple_score)).flatten()
                d.sampled_tuple = rng.choice(np.argwhere(tuple_score == np.amax(tuple_score)).flatten())
                sampled_index = np.random.choice(max_args)
                sampled_tuples.append(sampled_index)
                tuple_score = np.delete(tuple_score, sampled_index)
        if not self.VERBOSE:
            for sampled_tuple in sampled_tuples:
                # Display the errors and request user input for corrections
                row_data = d.dataframe.iloc[sampled_tuple, :]
                print(f"Row {sampled_tuple}: {row_data}")
                corrected_row = []
                for col, value in enumerate(row_data):
                    if (sampled_tuple, col) in d.detected_cells and value != '':
                        print(f"  Column {col}, Detected Error: {value}")
                        correction = input(f"    Please provide a correction for this value: ")
                        corrected_row.append(correction)
                    else:
                        corrected_row.append(value)
                d.labeled_cells[sampled_tuple] = corrected_row
                self.sampled_tuples += 1
        else:
            for sampled_tuple in sampled_tuples:
                self.label_with_ground_truth(d, sampled_tuple)
                self.sampled_tuples += 1

        self.logger.debug('Finish tuple sampling.')

    def label_with_ground_truth(self, d, sampled_tuple):
        """
        This method labels a tuple with ground truth.
        Takes a sampled row from sample_tuple(), iterates over each cell
        in that row taken from the clean data, and then adds
        d.labeled_cells[(row, col)] = [is_error, clean_value_from_clean_dataframe]
        to d.labeled_cells.
        """
        d.labeled_tuples[sampled_tuple] = 1
        for col in range(d.dataframe.shape[1]):
            cell = (sampled_tuple, col)
            error_label = 0
            if d.dataframe.iloc[cell] != d.clean_dataframe.iloc[cell]:
                error_label = 1
            d.labeled_cells[cell] = [error_label, d.clean_dataframe.iloc[cell]]
        self.logger.debug(f'Finished labeling tuple {sampled_tuple} with ground truth.')

    def update_models(self, d):
        """
        This method updates Baran's error corrector models with a new labeled tuple.
        """
        self.logger.debug('Start updating models.')
        cleaned_sampled_tuple = []
        for sampled_tuple in d.labeled_tuples:
            for column in range(d.dataframe.shape[1]):
                clean_cell = d.labeled_cells[(sampled_tuple, column)][1]
                cleaned_sampled_tuple.append(clean_cell)

            for column in range(d.dataframe.shape[1]):
                cell = (sampled_tuple, column)
                update_dictionary = {
                    "column": column,
                    "old_value": d.dataframe.iloc[cell],
                    "new_value": cleaned_sampled_tuple[column],
                }

                # if the value in that cell has been labeled an error
                if d.labeled_cells[cell][0] == 1:  # update value models.
                    if 'value' in self.FEATURE_GENERATORS:
                        self._value_based_models_updater(d.value_models, update_dictionary)

                    # if the cell hadn't been detected as an error
                    if cell not in d.detected_cells:
                        # add that cell to detected_cells and assign it IGNORE_SIGN
                        # --> das passiert, wenn die Error Detection nicht perfekt
                        # war, dass man einen Fehler labelt, der vorher noch nicht
                        # gelabelt war.
                        d.detected_cells[cell] = self.IGNORE_SIGN
            logging.debug('Finish updating models.')

    def _feature_generator_process(self, args):
        """
        This method supports Baran correctors. It generates cleaning suggestions for one error
        in one cell. The suggestion gets turned into features for the classifier in
        predict_corrections(). It gets called once for each error cell.
        """
        d, error_cell, is_synth = args

        # vicinity ist die Zeile, column ist die Zeilennummer, old_value ist der Fehler
        error_dictionary = {"column": error_cell[1],
                            "old_value": d.dataframe.iloc[error_cell],
                            "vicinity": list(d.dataframe.iloc[error_cell[0], :]),
                            "row": error_cell[0]}

        if "vicinity" in self.FEATURE_GENERATORS:
            if self.VICINITY_FEATURE_GENERATOR == 'naive':
                for o in self.VICINITY_ORDERS:
                    naive_corrections = pdep.vicinity_based_corrector_order_n(
                        counts_dict=d.vicinity_models[o],
                        ed=error_dictionary)
                    if is_synth:
                        for lhs in naive_corrections:
                            d.inferred_corrections.get(f'vicinity_{o}_{str(lhs)}')[error_cell] = naive_corrections[lhs]
                    else:
                        for lhs in naive_corrections:
                            d.corrections.get(f'vicinity_{o}_{str(lhs)}')[error_cell] = naive_corrections[lhs]

            elif self.VICINITY_FEATURE_GENERATOR == 'pdep':
                for o in self.VICINITY_ORDERS:
                    pdep_corrections = pdep.pdep_vicinity_based_corrector(
                        inverse_sorted_gpdeps=d.inv_vicinity_gpdeps[o],
                        counts_dict=d.vicinity_models[o],
                        ed=error_dictionary,
                        n_best_pdeps=self.N_BEST_PDEPS,
                        features_selection=self.PDEP_FEATURES,
                        gpdep_threshold=self.GPDEP_THRESHOLD)
                    if is_synth:
                        d.inferred_corrections.get(f'vicinity_{o}')[error_cell] = pdep_corrections
                    else:
                        d.corrections.get(f'vicinity_{o}')[error_cell] = pdep_corrections
            else:
                raise ValueError(f'Unknown VICINITY_FEATURE_GENERATOR '
                                 f'{self.VICINITY_FEATURE_GENERATOR}')

        if "value" in self.FEATURE_GENERATORS:
            if not is_synth:
                value_correction_suggestions = self._value_based_corrector(d.value_models, error_dictionary)
                for model_name, encodings_list in value_correction_suggestions.items():
                    for encoding, correction_suggestions in zip(['identity', 'unicode'], encodings_list):
                        d.corrections.get(f'value_{model_name}_{encoding}')[error_cell] = correction_suggestions

    def draw_synth_error_positions(self, d):
        """
        Draw the positions of synthetic missing values used to gain additional training data.
        """
        d.synthetic_error_cells = []
        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)
        row_errors = error_positions.original_row_errors()

        # determine error-free rows to sample from.
        candidate_rows = [(row, len(cells)) for row, cells in row_errors.items() if len(cells) == 0]
        ranked_candidate_rows = sorted(candidate_rows, key=lambda x: x[1])

        if self.SYNTH_TUPLES > 0 and len(ranked_candidate_rows) > 0:
            if len(ranked_candidate_rows) >= self.SYNTH_TUPLES:  # more candidate rows available than required, sample needed amount
                synthetic_error_rows = random.sample([x[0] for x in ranked_candidate_rows], self.SYNTH_TUPLES)
            else:  # less clean rows available than inferred tuples requested, take all you get.
                logging.info(
                    f'Requested {self.SYNTH_TUPLES} tuples to inferr features from, but only {len(ranked_candidate_rows)} error-free tuples are available.')
                logging.info(f'Sampling {len(ranked_candidate_rows)} rows instead.')
                synthetic_error_rows = [x[0] for x in ranked_candidate_rows]
            d.synthetic_error_cells = [(i, j) for i in synthetic_error_rows for j in range(d.dataframe.shape[1])]

    def prepare_augmented_models(self, d, synchronous=False):
        """
        Prepare Mimir's augmented models:
        1) Calculate gpdeps and append them to d.
        2) Train auto_instance model for each column.
        """

        n_workers = min(multiprocessing.cpu_count() - 1, 24)
        shape = d.dataframe.shape
        error_positions = helpers.ErrorPositions(d.detected_cells, shape, d.labeled_cells)
        row_errors = error_positions.updated_row_errors()
        column_corrected_list: list[int] = []
        column_error_list: list[tuple[int, int]] = []
        columns_for_llm_transformation: list = []
        server_process, session = self.start_llama_server(llm_name=self.LLM_NAME_CORRFM, gpu_layers=self.LLM_GPU_LAYERS,
                                                          ctx_size=self.LLM_CTX_SIZE, temp=self.LLM_TEMP,
                                                          top_k=self.LLM_TOP_K)

        if self.VICINITY_FEATURE_GENERATOR == 'pdep' and 'vicinity' in self.FEATURE_GENERATORS:
            d.inv_vicinity_gpdeps = {}
            for order in self.VICINITY_ORDERS:
                vicinity_gpdeps = pdep.calc_all_gpdeps(d.vicinity_models, d.lhs_values_frequencies, shape, row_errors,
                                                       order)
                d.inv_vicinity_gpdeps[order] = pdep.invert_and_sort_gpdeps(vicinity_gpdeps)

        if 'fd' in self.FEATURE_GENERATORS:
            self.logger.debug('Start FD profiling.')
            # calculate FDs
            inputted_rows = list(d.labeled_tuples.keys())
            df_user_input = d.clean_dataframe.iloc[inputted_rows, :]  # careful, this is ground truth.
            df_clean_iterative = pdep.cleanest_version(d.dataframe, df_user_input)
            d.fds = pdep.mine_fds(df_clean_iterative, d.clean_dataframe)
            self.logger.debug('Profiled FDs.')

            # calculate gpdeps
            shape = d.dataframe.shape
            error_positions = helpers.ErrorPositions(d.detected_cells, shape, d.labeled_cells)
            row_errors = error_positions.updated_row_errors()
            self.logger.debug('Calculated error positions.')

            d.fd_counts_dict, lhs_values_frequencies = pdep.fast_fd_counts(d.dataframe, row_errors, d.fds)
            self.logger.debug('Mined FD counts.')

            gpdeps = pdep.fd_calc_gpdeps(d.fd_counts_dict, lhs_values_frequencies, shape, row_errors, synchronous)
            self.logger.debug('Calculated gpdeps.')

            d.fd_inverted_gpdeps = {}
            for lhs in gpdeps:
                for rhs in gpdeps[lhs]:
                    if rhs not in d.fd_inverted_gpdeps:
                        d.fd_inverted_gpdeps[rhs] = {}
                    d.fd_inverted_gpdeps[rhs][lhs] = gpdeps[lhs][rhs]

            # normalize gpdeps per rhs
            for rhs in d.fd_inverted_gpdeps:
                norm_sum = 0
                for lhs, pdep_tuple in d.fd_inverted_gpdeps[rhs].items():
                    if pdep_tuple is not None:
                        norm_sum += pdep_tuple.gpdep
                if norm_sum > 0:
                    for lhs, pdep_tuple in d.fd_inverted_gpdeps[rhs].items():
                        if pdep_tuple is not None:
                            d.fd_inverted_gpdeps[rhs][lhs] = pdep.PdepTuple(pdep_tuple.pdep,
                                                                            pdep_tuple.gpdep,
                                                                            pdep_tuple.epdep,
                                                                            pdep_tuple.gpdep / norm_sum)
            # self.logger.debug('')

        if 'llm_master' in self.FEATURE_GENERATORS:
            # TODO entfernen
            start_time = time.time()


            fetch_llm_master_args = []

            # Prepare clean data subset for context
            inputted_rows = list(d.labeled_tuples.keys())
            user_input = d.clean_dataframe.iloc[inputted_rows, :]
            df_clean_subset = auto_instance.get_clean_table(d.dataframe, d.detected_cells, user_input)
            error_free_rows = df_clean_subset.shape[0]

            df_error_free_subset = df_clean_subset.sample(min(100, error_free_rows))

            # Construct prompts for each detected cell
            for (row, col) in d.detected_cells:
                old_value = d.dataframe.iloc[(row, col)]
                if old_value != '':
                    if helpers.fetch_cache(d.name, (row, col), 'llm_master', d.error_fraction, d.version, d.error_class,
                                           self.LLM_NAME_CORRFM) is None:
                        df_row_with_error = d.dataframe.iloc[row, :].copy()
                        prompt = helpers.llm_master_prompt((row, col), df_error_free_subset, df_row_with_error)
                        fetch_llm_master_args.append(
                            [prompt, old_value, d.name, (row, col), 'llm_master', session, d.error_fraction,
                             d.version, d.error_class, self.LLM_NAME_CORRFM]
                        )

            self.logger.debug(
                f'Identified {len(fetch_llm_master_args)} llm_master corrections not yet cached. Fetching...')

            if len(fetch_llm_master_args) > 0:
                if synchronous:
                    llm_master_results = map(helpers.generate_llm_response, *zip(*fetch_llm_master_args))
                else:
                    chunksize = len(fetch_llm_master_args) // min(len(fetch_llm_master_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        llm_master_results = executor.map(helpers.generate_llm_response, *zip(*fetch_llm_master_args),
                                                          chunksize=chunksize)
                for llm_result in llm_master_results:
                    if llm_result is not None:
                        helpers.insert_llm_into_cache(llm_result)
                    else:
                        self.logger.debug('Unable to fetch llm_master result.')
            self.logger.debug(
                f'Fetched {len(fetch_llm_master_args)} llm_master corrections and added them to the cache.')

            # TODO entfernen
            end_time = time.time()
            self.time_llm_master = end_time - start_time
            self.llm_master_errors = len(fetch_llm_master_args)
            self.llm_master_errors_corrected = len(fetch_llm_master_args)

        if 'llm_transformation' in self.FEATURE_GENERATORS:  # try to find a Transformation for each column with a llm
            # TODO entfernen
            start_time = time.time()

            error_correction_pairs: dict[int, List[Tuple[str, str]]] = {}
            # step 1:
            # Construct pairs of ('error', 'correction') per column by iterating over the user input.
            for cell in d.labeled_cells:
                if cell in d.detected_cells:
                    error = d.detected_cells[cell]
                    correction = d.labeled_cells[cell][1]
                    if error != '':
                        if correction == '':  # encode missing value
                            correction = '<MV>'
                        if error_correction_pairs.get(cell[1]) is None:
                            error_correction_pairs[cell[1]] = []
                        error_correction_pairs[cell[1]].append((error, correction))

            # Count errors per column
            error_counts: dict[int, int] = {}
            for (row, col), error in d.detected_cells.items():
                error_counts[col] = error_counts.get(col, 0) + 1

            # check with user corrections, if the llm can find a valid correction (> 60%), if yes the transformation analyses starts
            failed_pairs_dict: dict[int, list[tuple[str, str]]] = {}
            for column in error_correction_pairs:
                if len(error_correction_pairs.get(column)) >= 5 and error_counts[
                    column] > 200 and not helpers.is_column_in_cache(d.name, column, 'llm_correction',
                                                                     self.LLM_NAME_CORRFM):
                    column_name, category = d.column_categories[column]
                    correction_quota, failed_pairs = helpers.llm_test_user_corrections(self.LLM_NAME_CORRFM, session,
                                                                                       error_correction_pairs.get(
                                                                                           column), column_name,
                                                                                       category)
                    if correction_quota >= 0.6:
                        columns_for_llm_transformation.append(column)
                        if column not in failed_pairs_dict:
                            failed_pairs_dict[column] = []
                        if len(failed_pairs) > 0:
                            for pair in failed_pairs:
                                failed_pairs_dict[column].append(pair)

            # if a column has more than 75% of all errors, the transformation analysis will be tried anyway
            for column in error_counts:
                error_count = error_counts[column]
                if (error_count / len(d.detected_cells)) >= 0.75:
                    if column not in columns_for_llm_transformation and not helpers.is_column_in_cache(d.name, column,
                                                                                                       'llm_correction',
                                                                                                       self.LLM_NAME_CORRFM):
                        columns_for_llm_transformation.append(column)

            self.logger.debug(f"Identified {len(columns_for_llm_transformation)} columns for llm_transformation.")

            if len(columns_for_llm_transformation) > 0:
                self.stop_llama_server(server_process)
                server_process, session = self.start_lmstudio_server()

            for col in columns_for_llm_transformation:
                self.logger.debug(f"beginning with llm_transformation for column [{col}]...")
                iterations: int = 0
                quota: float = 0
                is_revision: bool = False
                last_try: bool = False
                adjusted_pairs: list[tuple[str, str]] = []
                initial_response: str = ""
                response: str = ""
                initial_method_code: str = ""
                initial_failed_tests: dict = {}
                pairs: list[tuple[str, str]] | None = error_correction_pairs.get(col)
                column_name, column_category = d.column_categories[col]
                while quota < 1 and iterations < 6:

                    if iterations == 4 and self.LAST_TRY_THRESHOLD < 1.0:
                        self.logger.debug(
                            f"didnt find a transformation yet, trying with last_try and a threshold of {self.LAST_TRY_THRESHOLD}")
                        last_try = True
                        if len(failed_pairs_dict[col]) == 0:
                            break
                        adjusted_pairs = [pair for pair in pairs if pair not in failed_pairs_dict[
                            col]]  # adjusted_pair are all pairs which the llm found a correct correction for
                        if not (len(adjusted_pairs) / len(pairs) >= self.LAST_TRY_THRESHOLD and len(
                                adjusted_pairs) > 4):  # there must be at least 'threshold' pairs and at least four pairs left
                            break

                    if iterations == 5 and not last_try:
                        break

                    if is_revision:
                        self.logger.debug(f"starting revision with last_try={last_try}")

                        self.logger.debug("step 4 - generating new method based on failed test cases...")
                        if last_try:
                            method_code, quota, failed_tests, generated_method = helpers.llm_generate_improved_method_failed_tests(
                                initial_response, initial_failed_tests, initial_method_code, adjusted_pairs)
                            self.logger.debug(
                                f"method failed in {len(failed_tests)} of {len(adjusted_pairs)} test cases")
                        else:
                            method_code, quota, failed_tests, generated_method = helpers.llm_generate_improved_method_failed_tests(
                                initial_response, initial_failed_tests, initial_method_code, pairs)
                            self.logger.debug(f"method failed in {len(failed_tests)} of {len(pairs)} test cases")
                        self.logger.debug(
                            f"step 4 - done generating new method based on failed test cases with last_try={last_try}")

                        if quota == 1:
                            self.logger.debug("no failed test cases - jumping to step 6")
                            break

                        self.logger.debug("step 5 - revision the given transformation...")
                        i = 0
                        while response == "" and i < 3:
                            response = helpers.llm_recognize_method_error(initial_response, initial_method_code,
                                                                          initial_failed_tests)
                            i += 1
                        self.logger.debug("step 5- done revision the given transformation")

                        if response == "":
                            break

                        self.logger.debug(f"step 6 - generating method from revision with last_try={last_try}...")
                        method_code = ""
                        i = 0
                        while (method_code == "" or len(failed_tests) != 0) and i < 3:
                            if last_try:
                                method_code, quota, failed_tests, generated_method = helpers.llm_generate_improved_method_revision(
                                    initial_response, response, initial_method_code, adjusted_pairs)
                                self.logger.debug(
                                    f"method failed in {len(failed_tests)} of {len(adjusted_pairs)} test cases")
                            else:
                                method_code, quota, failed_tests, generated_method = helpers.llm_generate_improved_method_revision(
                                    initial_response, response, initial_method_code, pairs)
                                self.logger.debug(
                                    f"method failed in {len(failed_tests)} of {len(pairs)} test cases")
                            i += 1
                        self.logger.debug(
                            f"step 6 - done generating method from revision with last_try={last_try}")

                        is_revision = False
                        last_try = False
                        initial_response = ""
                        initial_method_code = ""
                        response = ""
                        adjusted_pairs = []
                    else:

                        self.logger.debug("step 2 - analysing transformation...")
                        i = 0
                        while initial_response == "" and i < 3:
                            if last_try:
                                initial_response = helpers.llm_recognize_correction(column_name, column_category,
                                                                                    adjusted_pairs)
                            else:
                                initial_response = helpers.llm_recognize_correction(column_name, column_category, pairs)
                            i += 1
                        if initial_response == "":
                            break
                        self.logger.debug(f"step 2 - done analysing transformation with last_try={last_try}:")

                        self.logger.debug(f"step 3 - generating method from transformation with last_try={last_try}...")
                        i = 0
                        while initial_method_code == "" and i < 3:
                            if last_try:
                                initial_method_code, quota, initial_failed_tests, generated_method = helpers.llm_generate_method(
                                    initial_response, adjusted_pairs)
                                self.logger.debug(
                                    f"method failed in {len(initial_failed_tests)} of {len(adjusted_pairs)} test cases")
                            else:
                                initial_method_code, quota, initial_failed_tests, generated_method = helpers.llm_generate_method(
                                    initial_response, pairs)
                                self.logger.debug(
                                    f"method failed in {len(initial_failed_tests)} of {len(pairs)} test cases")
                            i += 1
                        self.logger.debug(
                            f"step 3 - done generating method from transformation with last_try={last_try}")

                        is_revision = True

                    iterations += 1

                if quota == 1:
                    self.logger.debug(f"found transformation for column [{col}]")
                    self.logger.debug(" step 7 - applying correction for every error in this column...")
                    for (row, column) in d.detected_cells:
                        if column == col:
                            try:
                                result = generated_method(d.dataframe.iloc[(row, column)])
                                result_str: str = str(result)
                                llm_result = (
                                    helpers.prepare_correction_for_cache(d.name, (row, column),
                                                                         'llm_correction',
                                                                         result_str,
                                                                         d.error_fraction, d.version, d.error_class,
                                                                         self.LLM_NAME_CORRFM))
                                helpers.insert_llm_into_cache(llm_result)
                                if col not in column_corrected_list:
                                    column_corrected_list.append(col)
                            except Exception as e:
                                column_error_list.append((row, column))
                                self.logger.debug(e)
                    self.logger.debug(
                        f"step 7 - applied correction for {error_counts[col] - len(column_error_list)} of {error_counts[col]} errors in column [{col}]")

            total_corrections: int = 0
            total_errors: int = len(d.detected_cells)
            for column in column_corrected_list:
                total_corrections += error_counts[column]
            total_corrections -= len(column_error_list)
            if total_corrections > 0:
                self.logger.debug(f"Corrected {total_corrections} of {total_errors} with llm_transformation.")
            if len(columns_for_llm_transformation) > 0:
                self.stop_lmstudio_server(server_process)

            # TODO entfernen
            end_time = time.time()
            self.time_llm_transformation = end_time - start_time
            self.llm_transformation_errors = total_errors
            self.llm_transformation_errors_corrected = total_corrections
            self.llm_transformation_columns = len(columns_for_llm_transformation)

        if 'llm_correction' in self.FEATURE_GENERATORS:  # send all LLM-queries and add the to cache
            # TODO entfernen
            start_time = time.time()

            error_correction_pairs: dict[int, List[Tuple[str, str]]] = {}
            fetch_llm_correction_args = []
            if column_corrected_list is None:
                column_corrected_list: list[int] = []
            if len(columns_for_llm_transformation) > 0:
                server_process, session = self.start_llama_server(llm_name=self.LLM_NAME_CORRFM,
                                                                  gpu_layers=self.LLM_GPU_LAYERS,
                                                                  ctx_size=self.LLM_CTX_SIZE, temp=self.LLM_TEMP,
                                                                  top_k=self.LLM_TOP_K)

            # Construct pairs of ('error', 'correction') per column by iterating over the user input.
            for cell in d.labeled_cells:
                if cell in d.detected_cells:
                    error = d.detected_cells[cell]
                    correction = d.labeled_cells[cell][1]
                    if error != '':
                        if correction == '':  # encode missing value
                            correction = '<MV>'
                        if error_correction_pairs.get(cell[1]) is None:
                            error_correction_pairs[cell[1]] = []
                        error_correction_pairs[cell[1]].append((error, correction))

            for (row, col) in d.detected_cells:
                if col not in column_corrected_list or (row, col) in column_error_list:
                    column_name, category = d.column_categories[col]
                    old_value = d.dataframe.iloc[(row, col)]
                    if old_value != '' and error_correction_pairs.get(
                            col) is not None:  # Skip if there is no value to be transformed or no cleaning examples
                        if helpers.fetch_cache(d.name, (row, col), 'llm_correction', d.error_fraction, d.version,
                                               d.error_class, self.LLM_NAME_CORRFM) is None:
                            prompt = helpers.llm_correction_prompt(old_value, error_correction_pairs[col], column_name,
                                                                   category)
                            fetch_llm_correction_args.append(
                                [prompt, old_value, d.name, (row, col), 'llm_correction', session, d.error_fraction,
                                 d.version, d.error_class, self.LLM_NAME_CORRFM]
                            )

            self.logger.debug(
                f'Identified {len(fetch_llm_correction_args)} llm_correction corrections not yet cached. Fetching...')

            if len(fetch_llm_correction_args) > 0:
                if synchronous:
                    llm_results = map(helpers.generate_llm_response, *zip(*fetch_llm_correction_args))
                else:
                    chunksize = len(fetch_llm_correction_args) // min(len(fetch_llm_correction_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        llm_results = executor.map(helpers.generate_llm_response, *zip(*fetch_llm_correction_args),
                                                   chunksize=chunksize)

                for llm_result in llm_results:
                    if llm_result is not None:
                        helpers.insert_llm_into_cache(llm_result)
                    else:
                        self.logger.debug('Unable to fetch llm_correction result.')
                self.logger.debug(
                    f'Fetched {len(fetch_llm_correction_args)} llm_correction corrections and added them to the cache.')
            self.stop_llama_server(server_process)

            # TODO entfernen
            end_time = time.time()
            self.time_llm_correction = end_time - start_time
            self.llm_correction_errors = len(fetch_llm_correction_args)
            self.llm_correction_errors_corrected = len(fetch_llm_correction_args)

        if 'auto_instance' in self.FEATURE_GENERATORS:
            self.logger.debug('Start training DataWig Models.')

            # Simulate user input by reading labeled data from the typed dataframe
            inputted_rows = list(d.labeled_tuples.keys())
            typed_user_input = d.typed_clean_dataframe.iloc[inputted_rows, :]
            df_clean_subset = auto_instance.get_clean_table(d.typed_dataframe, d.detected_cells, typed_user_input)

            # Model training is very expensive. Train models only for columns that contain errors.
            error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)
            column_errors = error_positions.original_column_errors()
            columns_with_errors = [c for c in column_errors if len(column_errors[c]) > 0]

            columns = [(i_col, col) for i_col, col in enumerate(df_clean_subset.columns) if
                       i_col in columns_with_errors]
            for i_col, col in columns:
                imp = auto_instance.train_cleaning_model(df_clean_subset,
                                                         d.name,
                                                         label=i_col,
                                                         time_limit=self.TRAINING_TIME_LIMIT,
                                                         use_cache=self.AUTO_INSTANCE_CACHE_MODEL)
                if imp is not None:
                    # only infer rows that contain an error in column i_col or rows that contain a synthetic error_cell
                    # this saves a lot of runtime for large datasets with few errors.
                    rows_to_predict = [x[0] for x in column_errors[i_col]] + [x[0] for x in d.synthetic_error_cells if
                                                                              x[1] == i_col]
                    self.logger.debug(f'Trained DataWig model for column {col} ({i_col}).')
                    try:
                        d.imputer_models[i_col] = imp.predict_proba(d.typed_dataframe.iloc[rows_to_predict, :])
                        self.logger.debug(f'Used DataWig model to infer values for column {col} ({i_col}).')
                    except Exception as e:
                        print(f'Failed to infer values with DataWig model for column {col} ({i_col}):')
                        print(e)
                else:
                    d.imputer_models[i_col] = None
                    self.logger.debug(f'Failed to train a DataWig model for column {col} ({i_col}).')

    def generate_features(self, d, synchronous):
        """
        Use correctors to generate correction features.
        """
        process_args_list = [[d, cell, False] for cell in d.detected_cells]
        n_workers = min(multiprocessing.cpu_count() - 1, 24)

        self.logger.debug('Start user feature generation of Mimir Correctors.')

        if "fd" in self.FEATURE_GENERATORS:
            fd_pdep_args = []
            fd_results = []
            for row, col in d.detected_cells:
                gpdeps = d.fd_inverted_gpdeps.get(col)
                if gpdeps is not None:
                    local_counts_dict = {lhs_cols: d.fd_counts_dict[lhs_cols] for lhs_cols in
                                         gpdeps}  # save memory by subsetting counts_dict
                    row_values = list(d.dataframe.iloc[row, :])
                    fd_pdep_args.append([(row, col), local_counts_dict, gpdeps, row_values, self.FD_FEATURE])

            if len(fd_pdep_args) > 0:
                if synchronous:
                    fd_results = map(correctors.generate_pdep_features, *zip(*fd_pdep_args))
                else:
                    chunksize = len(fd_pdep_args) // min(len(fd_pdep_args),
                                                         n_workers)  # makes it so that chunksize >= 0.
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        fd_results = executor.map(correctors.generate_pdep_features, *zip(*fd_pdep_args),
                                                  chunksize=chunksize)

            for r in fd_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating pdep-fd features.')

        if 'llm_correction' in self.FEATURE_GENERATORS:
            llm_correction_args = []
            llm_correction_results = []

            for (row, col) in d.detected_cells:
                llm_correction_args.append(
                    [(row, col), d.name, d.error_fraction, d.version, d.error_class, self.LLM_NAME_CORRFM])

            if len(llm_correction_args) > 0:
                if synchronous:
                    llm_correction_results = map(correctors.generate_llm_correction_features,
                                                 *zip(*llm_correction_args))
                else:
                    chunksize = len(llm_correction_args) // min(len(llm_correction_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        llm_correction_results = executor.map(correctors.generate_llm_correction_features,
                                                              *zip(*llm_correction_args), chunksize=chunksize)

            for r in llm_correction_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating llm-correction features.')

        if 'llm_master' in self.FEATURE_GENERATORS:
            # use large language model to correct an error based on the error's vicinity. Inspired by Narayan et al.
            # 2022.
            llm_master_args = []
            llm_master_results = []

            for (row, col) in d.detected_cells:
                llm_master_args.append(
                    [(row, col), d.name, d.error_fraction, d.version, d.error_class, 'gpt-3.5-turbo'])

            if len(llm_master_args) > 0:
                if synchronous:
                    llm_master_results = map(correctors.generate_llm_master_features, *zip(*llm_master_args))
                else:
                    chunksize = len(llm_master_args) // min(len(llm_master_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        llm_master_results = executor.map(correctors.generate_llm_master_features,
                                                          *zip(*llm_master_args), chunksize=chunksize)

            for r in llm_master_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating llm-master features.')

        if 'auto_instance' in self.FEATURE_GENERATORS:
            auto_instance_args = []
            datawig_results = []
            for (row, col) in d.detected_cells:
                df_probas = d.imputer_models.get(col)
                if df_probas is not None:
                    auto_instance_args.append([(row, col), df_probas.loc[row], d.dataframe.iloc[row, col]])
            if len(auto_instance_args) > 0:
                if synchronous:
                    datawig_results = map(correctors.generate_datawig_features, *zip(*auto_instance_args))
                else:
                    chunksize = len(auto_instance_args) // min(len(auto_instance_args), n_workers)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        datawig_results = executor.map(correctors.generate_datawig_features, *zip(*auto_instance_args),
                                                       chunksize=chunksize)

            for r in datawig_results:
                d.corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating DataWig features.')

        # generate features Baran-style
        self.logger.debug('Start user feature generation of Baran Correctors.')
        for args in process_args_list:
            self._feature_generator_process(args)
        self.logger.debug('Finish user feature generation.')

        if self.VERBOSE:
            self.logger.info("User Features Generated.")

    def generate_inferred_features(self, d, synchronous):
        """
        Generate additional training data by using data from the dirty dataframe. This leverages the information about
        error positions, carefully avoiding additional training data that contains known errors.
        """
        synth_args_list = []
        n_workers = min(multiprocessing.cpu_count() - 1, 16)

        if self.SYNTH_TUPLES > 0 and len(d.synthetic_error_cells) > 0:
            synth_args_list = [[d, cell, True] for cell in d.synthetic_error_cells]

            self.logger.debug('Start inferred feature generation of Mimir Correctors.')

            if "fd" in self.FEATURE_GENERATORS:
                fd_pdep_args = []
                fd_results = []
                for row, col in d.synthetic_error_cells:
                    gpdeps = d.fd_inverted_gpdeps.get(col)
                    if gpdeps is not None:
                        local_counts_dict = {lhs_cols: d.fd_counts_dict[lhs_cols] for lhs_cols in
                                             gpdeps}  # save memory by subsetting counts_dict
                        row_values = list(d.dataframe.iloc[row, :])
                        fd_pdep_args.append([(row, col), local_counts_dict, gpdeps, row_values, self.FD_FEATURE])

                if len(fd_pdep_args) > 0:
                    if synchronous:
                        fd_results = map(correctors.generate_pdep_features, *zip(*fd_pdep_args))
                    else:
                        chunksize = len(fd_pdep_args) // min(len(fd_pdep_args), n_workers)
                        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                            fd_results = executor.map(correctors.generate_pdep_features, *zip(*fd_pdep_args),
                                                      chunksize=chunksize)

                for r in fd_results:
                    d.inferred_corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

            self.logger.debug('Finished generating inferred pdep-fd features.')

            if 'auto_instance' in self.FEATURE_GENERATORS:
                auto_instance_args = []
                datawig_results = []
                for (row, col) in d.synthetic_error_cells:
                    df_probas = d.imputer_models.get(col)
                    if df_probas is not None:
                        auto_instance_args.append([(row, col), df_probas.loc[row], ''])
                if len(auto_instance_args) > 0:
                    if synchronous:
                        datawig_results = map(correctors.generate_datawig_features, *zip(*auto_instance_args))
                    else:
                        chunksize = len(auto_instance_args) // min(len(auto_instance_args), n_workers)
                        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                            datawig_results = executor.map(correctors.generate_datawig_features,
                                                           *zip(*auto_instance_args), chunksize=chunksize)

                for r in datawig_results:
                    d.inferred_corrections.get(r['corrector'])[r['cell']] = r['correction_dict']

        self.logger.debug('Finished generating inferred DataWig features.')

        if len(synth_args_list) > 0:
            self.logger.debug('Start inferred feature generation of Baran Correctors.')
            for args in synth_args_list:
                self._feature_generator_process(args)
        self.logger.debug('Finish inferred feature generation.')

        # Stub to get llm_master to work in unsupervised setup.
        #
        # generate llm-features
        # for error_cell, model_name, prompt in ai_prompts:
        #     cache = helpers.fetch_cache(d.name,
        #                                 error_cell,
        #                                 model_name,
        #                                 d.error_fraction,
        #                                 d.version,
        #                                 d.error_class)
        #     if cache is not None:
        #         correction, token_logprobs, top_logprobs = cache
        #         correction_dicts = helpers.llm_response_to_corrections(correction, token_logprobs, top_logprobs)
        #         d.inferred_corrections.get(model_name)[error_cell] = correction_dicts

    def binary_predict_corrections(self, d):
        """
        The ML problem as formulated in the Baran paper.
        """
        self.logger.debug('Start training meta learner.')
        error_positions = helpers.ErrorPositions(d.detected_cells, d.dataframe.shape, d.labeled_cells)
        column_errors = error_positions.original_column_errors()
        columns_with_errors = [c for c in column_errors if len(column_errors[c]) > 0]
        pair_features = d.corrections.assemble_pair_features()
        synth_pair_features = d.inferred_corrections.assemble_pair_features()

        for j in columns_with_errors:
            if d.corrections.et_valid_corrections_made(d.corrected_cells,
                                                       j) > 0:  # ET model mentioned ground truth once in suggestions
                score = 0  # drop inferred_features to not distort correction classifier.

            else:  # evaluate synth tuples.
                score = ml_helpers.test_synth_data(d,
                                                   pair_features,
                                                   synth_pair_features,
                                                   self.CLASSIFICATION_MODEL,
                                                   j,
                                                   column_errors,
                                                   clean_on=self.TEST_SYNTH_DATA_DIRECTION)

            if score >= self.SYNTH_CLEANING_THRESHOLD:
                # now that we are certain about the synth data's usefulness, use additional training data.
                x_train, y_train, x_test, user_corrected_cells, error_correction_suggestions = ml_helpers.generate_train_test_data(
                    column_errors,
                    d.labeled_cells,
                    pair_features,
                    d.dataframe,
                    synth_pair_features,
                    j)
            else:  # score is below threshold, don't use additional training data
                x_train, y_train, x_test, user_corrected_cells, error_correction_suggestions = ml_helpers.generate_train_test_data(
                    column_errors,
                    d.labeled_cells,
                    pair_features,
                    d.dataframe,
                    {},
                    j)

            is_valid_problem, predicted_labels, predicted_probas = ml_helpers.handle_edge_cases(x_train, x_test,
                                                                                                y_train)

            if is_valid_problem:
                if self.CLASSIFICATION_MODEL == "ABC" or sum(y_train) <= 2:
                    gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                    gs_clf.fit(x_train, y_train, )
                elif self.CLASSIFICATION_MODEL == "CV":
                    gs_clf = hpo.cross_validated_estimator(x_train, y_train)
                else:
                    raise ValueError('Unknown model.')

                predicted_labels = gs_clf.predict(x_test)
                predicted_probas = [x[1] for x in gs_clf.predict_proba(x_test)]

            cells_without_suggestions = ml_helpers.set_binary_cleaning_suggestions(predicted_labels, predicted_probas,
                                                                                   x_test, error_correction_suggestions,
                                                                                   d.corrected_cells,
                                                                                   list(d.detected_cells.keys()))

            # for all errors that went uncorrected, take the sum of correction suggestions to determine a correciton.
            for cell in cells_without_suggestions:
                features = pair_features[cell]
                if len(features) > 0:
                    max_feature_correction = max(features, key=lambda k: sum(features[k]))
                    d.corrected_cells[cell] = max_feature_correction

        self.logger.debug('Finish inferring corrections.')

        if self.DATASET_ANALYSIS:
            samples = {c: random.sample(column_errors[c], min(40, len(column_errors[c]))) for c in column_errors}
            normalized_dataset = d.name
            normalized_dataset = f'{d.name}_{d.error_class}_{d.error_fraction}'

            analysis = {
                'dataset': normalized_dataset,
                'error_stats': [{'column': c, 'errors': len(e)} for c, e in column_errors.items()],
                'samples': [
                    {'column': col, 'cell': c, 'error': d.dataframe.iloc[c], 'correction': d.clean_dataframe.iloc[c],
                     'correction_suggestion': d.corrected_cells.get(c, 'no corrections suggested')}
                    for col in samples for c in samples[col]],
                'correctors': [{'corrector': cor,
                                'column': col,
                                'cell': c,
                                'error': d.dataframe.iloc[c],
                                'correction': d.clean_dataframe.iloc[c],
                                'correction_suggestions': d.corrections.correction_store[cor].get(c)} for cor in
                               d.corrections.correction_store for col in samples for c in samples[col]]
            }

            with open(f'analysis/{normalized_dataset}.json', 'wt') as f:
                json.dump(analysis, f)

        if self.VERBOSE:
            self.logger.info("{:.0f}% ({} / {}) of data errors are corrected.".format(
                100 * len(d.corrected_cells) / len(d.detected_cells),
                len(d.corrected_cells), len(d.detected_cells)))

    def clean_with_user_input(self, d):
        """
        User input ideally contains completely correct data. It should be leveraged for optimal cleaning
        performance.
        """
        self.logger.debug('Start cleaning with user input')
        if not self.CLEAN_WITH_USER_INPUT:
            return None
        for error_cell in d.detected_cells:
            if error_cell in d.labeled_cells:
                d.corrected_cells[error_cell] = d.labeled_cells[error_cell][1]
        self.logger.debug('Finish cleaning with user input')

    def store_results(self, d):
        """
        This method stores the results.
        """
        ec_folder_path = os.path.join(d.results_folder, "error-correction")
        if not os.path.exists(ec_folder_path):
            os.mkdir(ec_folder_path)
        pickle.dump(d, open(os.path.join(ec_folder_path, "correction.dataset"), "wb"))

    def start_lmstudio_server(self) -> Tuple[subprocess.Popen, requests.Session]:
        """
        Startet den LM Studio Server als Subprozess und gibt den Prozess sowie eine konfigurierte Requests-Session zurück.
        """

        # Prüfe, ob der Befehl "lms" verfügbar ist
        lms_executable = shutil.which("lms")
        if not lms_executable:
            self.logger.info("lms command not found, installing via npx lmstudio install-cli...")
            # Starte den Installationsbefehl
            install_cmd = "npx lmstudio install-cli"
            subprocess.run(install_cmd, check=True)
            # Erneut prüfen
            lms_executable = shutil.which("lms")
            if not lms_executable:
                raise FileNotFoundError("lms command not found after installation attempt.")

        # Baue den vollständigen Befehl zum Starten des Servers zusammen
        cmd = [lms_executable, "server", "start", "--port", "1234"]

        # Starte den Server als Subprozess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.logger.debug("Started LM Studio Server.")

        # Erstelle und konfiguriere eine Requests-Session
        session = requests.Session()
        session.headers.update({
            "Connection": "keep-alive"
        })

        # Unterdrücke Ausgaben von urllib3
        logging.getLogger("urllib3").setLevel(logging.ERROR)

        return process, session

    def stop_lmstudio_server(self, process: subprocess.Popen) -> None:
        """
        Stoppt den LM Studio Server, indem zuerst das geladene Modell entladen wird
        (mittels "lms unload <model_key>") und anschließend der Server gestoppt wird ("lms server stop").
        """
        # Prüfe, ob der Befehl 'lms' verfügbar ist
        lms_executable = shutil.which("lms")
        if not lms_executable:
            self.logger.error("lms command not found; cannot stop LM Studio Server.")
            raise ReferenceError("lms command not found; cannot stop LM Studio Server.")

        # lms ps um model_key zu erhalten
        try:
            result = subprocess.run(
                [lms_executable, "ps"],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running 'lms ps': {e}")
            raise Exception(f"Error running 'lms ps': {e}")

        # model_key rausholen
        model_key = None
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Identifier:"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    model_key = parts[1].strip()
                break

        if not model_key:
            self.logger.error("No model key found in 'lms ps' output.")
            raise Exception("No model key found in 'lms ps' output.")

        # lms unload model_key
        try:
            subprocess.run(
                [lms_executable, "unload", model_key],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error unloading model {model_key}: {e}")

        # lms server stop
        try:
            subprocess.run(
                [lms_executable, "server", "stop"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error stopping LM Studio Server: {e}")

    def start_llama_server(self, llm_name: str, gpu_layers: int, ctx_size: int, temp: int, top_k: int) \
            -> Tuple[subprocess.Popen, requests.Session]:
        """
        Startet den nativ gebauten llama.cpp-Server als Subprozess und gibt den Prozess sowie eine konfigurierte Requests-Session zurück..
        """

        # get the path of correction.py
        current_file = os.path.abspath(__file__)
        # go one folder up
        project_dir = os.path.dirname(os.path.dirname(current_file))
        # get paths of llama.cpp-server and models
        server_path: str = os.path.join(project_dir, "llama.cpp", "build", "bin", "llama-server")
        model_path: str = os.path.join(project_dir, "models", f"{llm_name}")

        # Kommandozeilenargumente für den Server
        cmd = [
            server_path,
            "-m", model_path,
            "--port", "8080",
            "--temp", f"{temp}",
            "--top-k", f"{top_k}",
            "--n-gpu-layers", f"{gpu_layers}",
            "--ctx-size", f"{ctx_size}",
        ]

        # Den Server starten und im Hintergrund laufen lassen
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.logger.debug("Started llama.cpp-Server.")

        # Kurze Pause, um dem Server Zeit zu geben, hochzufahren => sonst kommen nur Fehlermeldungen mit "model loading"
        time.sleep(2)
        session = requests.Session()
        session.headers.update({
            "Connection": "keep-alive"
        })

        # Unterdrücke Ausgaben von urllib3
        logging.getLogger("urllib3").setLevel(logging.ERROR)

        return process, session

    def stop_llama_server(self, process: subprocess.Popen) -> None:
        """
        Stoppt den laufenden llama.cpp-Server.
        """
        process.terminate()  # Signal zum Beenden senden
        process.wait()
        self.logger.debug("Stopped llama.cpp-Server.")

    def run(self, d, random_seed: int, synchronous: bool):
        """
        This method runs Mimir on an input dataset to correct data errors. A random_seed introduces
        some determinism in the feature_sampling process, but generally the cleaning process is
        non-deterministic despite a random_seed.
        The `synchronous` parameter, if set to true, will execute Mimir synchronously. If set to false,
        multiprocessing will be used to speed up the computation by shifting expensive operations
        concurrently onto multiple cores.
        """

        # This makes the sampling process deterministic.
        random.seed(random_seed)

        """
        Das alles in eigene Methode auslagern. Fall, dass kein modell passt, fehlt noch und logischerweise der wichtigste Schritt,
        den Fehlern einen Komplexitätsscore zu geben. 
        
        Vielleich ist auch das ganze vorher abgefrage sinnlos. Vielleicht sollte man einfach nur die Größe der GraKa abfragen oder halt
        irgendwie ermitteln und dann sogar mit mehreren LLMs arbeiten. Oder halt dann automatisch erkennen welches LLM funktioniert.
        Aber ganz ohne UserInput ists schlecht, der sollte schon entscheiden können ob er das beste oder schnellste oder einen automatischen
        mittelweg haben möchte
        """

        # TODO entfernen
        start_time = time.time()
        self.logger.setLevel(logging.INFO)

        d = self.initialize_dataset(d)
        self.categorize_columns(d)
        if len(d.detected_cells) == 0:
            raise ValueError('There are no errors in the data to correct.')
        if self.VERBOSE:
            self.logger.info(f"Start Mimir To Correct Dataset {d.name}\n")
        self.initialize_models(d)

        self.sample_tuple(d)
        self.update_models(d)

        self.draw_synth_error_positions(d)

        self.prepare_augmented_models(d, synchronous)
        self.generate_features(d, synchronous)
        self.generate_inferred_features(d, synchronous)
        self.binary_predict_corrections(d)
        self.clean_with_user_input(d)

        # TODO entfernen
        end_time = time.time()
        self.time_correction = end_time - start_time

        if self.VERBOSE:
            p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
            self.logger.info(
                "Cleaning performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}\n".format(d.name, p, r,
                                                                                                         f))

            # TODO entfernen
            self.logger.info(f"Time for llm_master: {self.time_llm_master}")
            self.logger.info(f"corrected {self.llm_master_errors_corrected} of {self.llm_master_errors}")
            self.logger.info(f"Time for llm_transformation: {self.time_llm_transformation}")
            self.logger.info(f"corrected {self.llm_transformation_columns} columns with llm_transformation")
            self.logger.info(f"that are {self.llm_transformation_errors_corrected} of {self.llm_transformation_errors}")
            self.logger.info(f"Time for llm_correction: {self.time_llm_correction}")
            self.logger.info(f"corrected {self.llm_correction_errors_corrected} of {self.llm_correction_errors}")
            self.logger.info(f"Time for correction: {self.time_correction}")

        return d.corrected_cells


# TODO am Ende wieder entfernen
def main(dataset_str: str, error_fraction_int: int, feature_generators: list, llm_name_corrfm: str):
    # store results for detailed analysis
    dataset_analysis = True

    dataset_name = dataset_str
    error_class = "simple_mcar"
    error_fraction = error_fraction_int
    version = 1
    n_rows = None

    labeling_budget = 20
    synth_tuples = 100
    synth_cleaning_threshold = 0.9
    auto_instance_cache_model = True
    clean_with_user_input = True  # Careful: If set to False, d.corrected_cells will remain empty.
    gpdep_threshold = 0.3
    training_time_limit = 90
    feature_generators = feature_generators
    last_try_threshold = 0.8  # if 1.0 last_try is off, if less than 1.0 it functions as the minimum precision needed for a valid transformation, less than 0.8 isn't recommended
    llm_temp = 0  # temp for the llm used in llm_master, llm_transformation, llm_correction
    llm_top_k = 1  # top_k for the llm used in llm_master, llm_transformation, llm_correction
    gpu_layers = 48  # gpu layers used by the llm on the gpu
    ctx_size = 4096  # size of the context window of the llm
    classification_model = "ABC"
    fd_feature = 'norm_gpdep'
    vicinity_orders = [1]
    n_best_pdeps = 3
    vicinity_feature_generator = "naive"
    pdep_features = ['pr']
    test_synth_data_direction = 'user_data'
    llm_name_corrfm = llm_name_corrfm  # for llm_transformation are two more llm's necessary: 'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf' and 'Codestral-22B-v0.1-Q2_K.gguf'
    sampling_technique = 'greedy'

    # Set this parameter to keep runtimes low when debugging
    data = dataset.Dataset(dataset_name, error_fraction, version, error_class, n_rows)
    data.detected_cells = data.get_errors_dictionary()

    logging.info(f'Initialized dataset {dataset_name}')

    app = Cleaning(labeling_budget, classification_model, clean_with_user_input, feature_generators, last_try_threshold,
                   llm_temp, llm_top_k, gpu_layers, ctx_size, vicinity_orders,
                   vicinity_feature_generator, auto_instance_cache_model, n_best_pdeps, training_time_limit,
                   synth_tuples, synth_cleaning_threshold, test_synth_data_direction, pdep_features, gpdep_threshold,
                   fd_feature, dataset_analysis, llm_name_corrfm, sampling_technique)
    app.VERBOSE = True  # also switches real user correction to simulated user correction with ground truth
    random_seed = 0
    correction_dictionary = app.run(data, random_seed,
                                    synchronous=True)  # when using llm_master, llm_transformation or llm_correction, synchronous must be True


if __name__ == "__main__":
    main("glass", 1, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
"""
    list_name_corrfm: list = ["Llama-3.2-3B-Instruct.fp16.gguf", "Llama-3.2-3B-Instruct.Q4_K_M.gguf", "Llama-3.2-3B-Instruct.IQ1_M.gguf"]

    for llm in list_name_corrfm:
        main("beers", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("flights", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("food", 3, ['llm_master', 'llm_correction', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("hospital", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("rayyan", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("tax", 3, ['llm_correction', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("bridges", 1, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("bridges", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("cars", 1, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("cars", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("glass", 1, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("glass", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("restaurant", 1, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("restaurant", 3, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("6", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("137", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("151", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("184", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("1481", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("41027", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
        i = 0
        while i < 4:
            print("\n")
            i += 1
        time.sleep(60)

        main("43572", 5, ['auto_instance', 'fd', 'llm_correction', 'llm_master', 'llm_transformation'], llm)
"""