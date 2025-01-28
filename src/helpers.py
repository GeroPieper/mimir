import json
import random
import sqlite3
import numpy as np
from typing import Union, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F

from transformers import GPT2TokenizerFast, GPT2LMHeadModel


# import openai
import pandas as pd
# import dotenv


# dotenv.load_dotenv()
# openai.api_key = os.environ.get('OPENAI_API_KEY')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")   # TODO anderes Modell einsetzen
model = GPT2LMHeadModel.from_pretrained("gpt2")

@dataclass
class LLMRequest:
    cell: Tuple[int, int]
    corrector_name: str
    prompt: Union[str, None]


@dataclass
class LLMResult:
    dataset: str
    row: int
    column: int
    correction_model_name: str
    correction_tokens: list
    token_logprobs: list
    top_logprobs: list
    error_fraction: Union[int, None]
    version: Union[int, None]
    error_class: Union[str, None]
    llm_name: str

@dataclass
class LLMResultGPT4All:
    prompt: str
    dataset: str
    row: int
    column: int
    response_text: str
    correction_model_name: str
    error_fraction: Union[int, None]
    version: Union[int, None]
    error_class: Union[str, None]
    llm_name: str


@dataclass
class ErrorPositions:
    detected_cells: Dict[Tuple[int, int], Union[str, float, int]]
    table_shape: Tuple[int, int]
    labeled_cells: Dict[Tuple[int, int], Tuple[int, str]]

    def original_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            column_errors[col].append((row, col))
        return column_errors

    @property
    def updated_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.labeled_cells:
                column_errors[col].append((row, col))
        return column_errors

    def original_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            row_errors[row].append((row, col))
        return row_errors

    def updated_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.labeled_cells:
                row_errors[row].append((row, col))
        return row_errors


class Corrections:
    """
    Store correction suggestions provided by the correction models in correction_store. In _feature_generator_process
    it is guaranteed that all models return something for each error cell -- if there are no corrections made, that
    will be an empty list. If a correction or multiple corrections has/have been made, there will be a list of
    correction suggestions and feature vectors.
    """

    def __init__(self, model_names: List[str]):
        self.correction_store = {name: dict() for name in model_names}

    def flat_correction_store(self):
        flat_store = {}
        for model in self.correction_store:
            flat_store[model] = self.correction_store[model]
        return flat_store

    @property
    def available_corrections(self) -> List[str]:
        return list(self.correction_store.keys())

    def features(self) -> List[str]:
        """Return a list describing the features the Corrections come from."""
        return list(self.correction_store.keys())

    def get(self, model_name: str) -> Dict:
        """
        For each error-cell, there will be a list of {corrections_suggestion: probability} returned here. If there is no
        correction made for that cell, the list will be empty.
        """
        return self.correction_store[model_name]

    def assemble_pair_features(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """Return features."""
        flat_corrections = self.flat_correction_store()
        pair_features = defaultdict(dict)
        for mi, model in enumerate(flat_corrections):
            for cell in flat_corrections[model]:
                for correction, pr in flat_corrections[model][cell].items():
                    # interessanter gedanke:
                    # if df_dirty.iloc[cell] == missing_value_token and model == 'llm_value':
                    #   pass
                    if correction not in pair_features[cell]:
                        features = list(flat_corrections.keys())
                        pair_features[cell][correction] = np.zeros(len(features))
                    pair_features[cell][correction][mi] = pr
        return pair_features

    def et_valid_corrections_made(self, corrected_cells: Dict[Tuple[int, int], str], column: int) -> int:
        """
        Per column, return how often the corrector leveaging error transformations mentioned the ground truth
        in its correction suggestions. The result is used to determine if ET models are useful to clean the column
        Depending on the outcome, the inferred_features are discarded.
        """
        if 'llm_correction' not in self.available_corrections or len(corrected_cells) == 0:
            return 0
        
        ground_truth_mentioned = 0
        for error_cell, correction in corrected_cells.items():
            if error_cell[1] == column:
                correction_suggestions = self.correction_store['llm_correction'].get(error_cell, {})
                if correction in list(correction_suggestions.keys()):
                    ground_truth_mentioned += 1
        return ground_truth_mentioned

class GPT4AllModelSession:
    _instance = None
    
    def __new__(cls, model):
        if cls._instance is None:
            cls._instance = super(GPT4AllModelSession, cls).__new__(cls)
            cls._instance.model = model
            cls._instance.chat_session = cls._instance.model.chat_session(system_prompt=
                "You are a data cleaning machine that detects patterns to return a correction. If you do "                                                                                                                                                         
                "not find a correction, you return the token <NULL>. You always follow the example and "
                "return NOTHING but the correction or <NULL>.\n---\n")
            print(f"Modell und Sitzung initialisiert. Mit {model.device}") #print zu logger Eintrag ändern
        return cls._instance
    
    def generate_response(self, prompt, max_tokens):
        #with model.chat_session:
            #return self.model.generate(prompt, max_tokens=max_tokens, temp=0.2)
        return self.model.generate(prompt, max_tokens=max_tokens, top_k=1, temp=0) # TODO Trade-Off von temp, top_k und Qualität, Laufzeit untersuchen

def connect_to_cache() -> sqlite3.Connection:
    """
    Connect to the cache for LLM prompts.
    @return: a connection to the sqlite3 cache.
    """
    conn = sqlite3.connect('cache.db')
    return conn


def fetch_cache(dataset: str,
                error_cell: Tuple[int,int],
                correction_model_name: str,
                error_fraction: Union[None, int] = None,
                version: Union[None, int] = None,
                error_class: Union[None, str] = None,
                llm_name: str = "gpt-3.5-turbo") -> Union[None, Tuple[dict, dict, dict]]:
    """
    Sending requests to LLMs is expensive (time & money). We use caching to mitigate that cost. As primary key for
    a correction serves (dataset_name, error_cell, version, correction_model_name, error_fraction, version, error_class, llm_name).
    This is imperfect, but a reasonable approximation: Since the prompt-generation itself as well as its dependencies are
    non-deterministic, the prompt cannot serve as part of the primary key.

    @param dataset: name of the dataset that is cleaned.
    @param error_cell: (row, column) position of the error.
    @param correction_model_name: "llm_master" or "llm_correction".
    @param error_fraction: Fraction of errors in the dataset.
    @param version: Version of the dataset. See dataset.py for details.
    @param error_class: Class of the error, e.g. MCAR
    @return: correction_tokens, token_logprobs, and top_logprobs.
    """
    dataset_name = dataset

    conn = connect_to_cache()
    cursor = conn.cursor()
    query = """SELECT
                 correction_tokens,
                 token_logprobs,
                 top_logprobs
               FROM cache
               WHERE
                 dataset=?
                 AND row=?
                 AND column=?
                 AND correction_model=?
                 AND llm_name=?"""
    parameters = [dataset_name, error_cell[0], error_cell[1], correction_model_name, llm_name]
    # Add conditions for optional parameters
    if error_fraction is not None:
        query += " AND error_fraction=?"
        parameters.append(error_fraction)
    else:
        query += " AND error_fraction IS NULL"

    if version is not None:
        query += " AND version=?"
        parameters.append(version)
    else:
        query += " AND version IS NULL"

    if error_class is not None:
        query += " AND error_class=?"
        parameters.append(error_class)
    else:
        query += " AND error_class IS NULL"

    cursor.execute(query, tuple(parameters))
    result = cursor.fetchone()
    conn.close()
    if result is not None:
        return json.loads(result[0]), json.loads(result[1]), json.loads(result[2])  # access the correction
    return None


def generate_llm_response(prompt: Union[str, None],
              old_value: str,
              dataset: str,
              error_cell: Tuple[int, int],
              correction_model_name: str,
              gpt_session: GPT4AllModelSession,
              error_fraction: Union[None, int] = None,
              version: Union[None, int] = None,
              error_class: Union[None, str] = None,
              #llm_name: str = "gpt-3.5-turbo"
              llm_name: str = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
              ) -> Union[LLMResultGPT4All, None]:
    """
    Generiert eine Antwort mit GPT4All.
    """
    try:
        response = gpt_session.generate_response(prompt, len(old_value))  # ist nur eine Annäherung an die benötigten Token
        response_text = response.split('\n', 1)[0]
        print(f"Error: {old_value}, Korrektur: {response_text}")

    except Exception as e:
        print(f"Error generating response: {e}")
        row, column = error_cell
        llm_result = LLMResultGPT4All(prompt, dataset, row, column, "NULL", correction_model_name,
                                      error_fraction, version, error_class, llm_name)
        return llm_result

    row, column = error_cell
    llm_result = LLMResultGPT4All(prompt, dataset, row, column, response_text, correction_model_name,
                           error_fraction, version, error_class, llm_name)
    return llm_result

def compute_token_logprobs(llm_result: LLMResultGPT4All) -> LLMResult | None:
    """
    Berechnet die Token-Logwahrscheinlichkeiten und Top-Logwahrscheinlichkeiten für eine Antwort.
    """
    try:
        # Tokenize Prompt und Response
        inputs = tokenizer(llm_result.prompt, return_tensors="pt")
        response_inputs = tokenizer(llm_result.response_text, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -response_inputs.input_ids.size(-1):, :]  # Nur relevante Logits
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs)

        # Extrahiere Token-Logwahrscheinlichkeiten
        correction_tokens = tokenizer.convert_ids_to_tokens(response_inputs.input_ids[0])
        token_logprobs = [
            log_probs[0, i, token_id].item() / 100000  # TODO: Skalierungsfaktor optimieren
            for i, token_id in enumerate(response_inputs.input_ids[0])
        ]

        # Berechne Top-Log-Wahrscheinlichkeiten
        top_logprobs = []
        """for i, token_id in enumerate(response_inputs.input_ids[0]):
            token_probs, token_indices = torch.topk(probs[0, i], k=5)
            top_logprobs.append({
                tokenizer.convert_ids_to_tokens(idx.item()): torch.log(prob).item()
                for prob, idx in zip(token_probs, token_indices)
            })"""

    except Exception as e:
        print(f"Error computing token logprobs: {e}")
        llm_result = LLMResult(llm_result.dataset, llm_result.row, llm_result.column, llm_result.correction_model_name,
                               ['NULL'], [], [], llm_result.error_fraction,
                               llm_result.version, llm_result.error_class, llm_result.llm_name)
        return llm_result

    llm_result = LLMResult(llm_result.dataset, llm_result.row, llm_result.column, llm_result.correction_model_name,
                           correction_tokens, token_logprobs, top_logprobs, llm_result.error_fraction,
                           llm_result.version, llm_result.error_class, llm_result.llm_name)

    return llm_result


def insert_llm_into_cache(llm_result: LLMResult):
    """
    Add a record to the cache if it isn't in there already.
    """
    conn = connect_to_cache()
    cursor = conn.cursor()

    # Check if a record with the same values already exists
    cursor.execute(
        """SELECT COUNT(*)
           FROM cache
           WHERE dataset = ? AND row = ? AND column = ? AND
                 correction_model = ? AND error_fraction = ? AND
                 version = ? AND error_class = ? AND llm_name = ?""",
        (llm_result.dataset, llm_result.row, llm_result.column, llm_result.correction_model_name, llm_result.error_fraction, llm_result.version, llm_result.error_class, llm_result.llm_name)
    )
    existing_records = cursor.fetchone()[0]
    
    # If no matching record found, insert the new record
    if existing_records == 0:
        cursor.execute(
            """INSERT INTO cache
               (dataset, row, column, correction_model, correction_tokens, token_logprobs, top_logprobs, error_fraction, version, error_class, llm_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (llm_result.dataset,
             llm_result.row,
             llm_result.column,
             llm_result.correction_model_name,
             json.dumps(llm_result.correction_tokens),
             json.dumps(llm_result.token_logprobs),
             json.dumps(llm_result.top_logprobs),
             llm_result.error_fraction,
             llm_result.version,
             llm_result.error_class,
             llm_result.llm_name)
        )
        conn.commit()
    else:
        print("Record already exists, skipping insertion.")

    conn.commit()
    conn.close()


def llm_response_to_corrections(correction_tokens: dict, token_logprobs: dict, top_logprobs: dict) -> Dict[str, float]:
    correction = ''.join(correction_tokens)
    correction = correction.replace('<MV>', '')  # parse missing value
    if correction.strip() in ['NULL', '<NULL>', 'null', '<null>']:
        return {}
    return {correction: np.exp(sum(token_logprobs))}


def error_free_row_to_prompt(df: pd.DataFrame, row: int, column: int) -> Tuple[str, str]:
    """
    Turn an error-free dataframe-row into a string, and replace the error-column with an <Error> token.
    Return a tuple of (stringified_row, correction). Be mindful that correction is only the correct value if
    the row does not contain an error to begin with.
    """
    if len(df.shape) == 1:  # final row, a series
        correction = ''
        values = df.values
    else:  # dataframe
        correction = df.iloc[row, column]
        values = df.iloc[row, :].values
    row_values = [f"{x}," if i != column else "<Error>," for i, x in enumerate(values)]
    assembled_row_values = ''.join(row_values)[:-1]
    return assembled_row_values, correction

def map_error_types(column_type):
    error_mapping = {
        "Numeric": [
            "Add Delta",
            "Outlier",
            "MissingValue",
            "WrongUnit",
            "Typo",
            "Replace",
            "CategorySwap",
            "WrongDataType"
        ],
        "String": [
            "Typo",
            "CategorySwap",
            "Mojibake",
            "Replace",
            "Extraneous",
            "Permute",
            "MissingValue",
            "WrongDataType"
        ],
        "DateTime": [
            "Typo",
            "Replace",
            "Outlier",
            "Add Delta",
            "Permute",
            "MissingValue",
            "WrongDataType",
            "WrongUnit",
            "FormatError"
        ],
        "Boolean": [
            "MissingValue",
            "Typo",
            "Replace",
            "WrongDataType",
            "CategorySwap"
        ],
    }
    return error_mapping.get(column_type, ["Unknown"])

def llm_correction_prompt(old_value: str, error_correction_pairs: List[Tuple[str, str]], column_name, category) -> str:
    """
    Generate the llm_correction prompt sent to the LLM.
    """

    prompt = ("You are a data cleaning machine that detects patterns to return a correction. If you do "
              "not find a correction, you return the token <NULL>. You always follow the example and "
              "return NOTHING but the correction, <MV> or <NULL>.\n---\n"
              f"column name: {column_name}\n"
              f"column category: {category}\n"
              f"possible errors for this category: {map_error_types(category)}\n"
              f"examples:\n---\n")


    """ "neuer" prompt ist schlechter als der "alte" - für später als Referenz aufheben
    prompt = ("You are a data cleaning machine designed to detect patterns and return corrections. If no correction is "   
              "found, respond with the token <NULL>.  Always adhere to the provided format and return ONLY the correction "
              "or <NULL>.")
    """
    error, correction = error_correction_pairs[0]
    # Von 10 auf 4 gekürzt um eine schnellere Verarbeitung zu erzielen # TODO Trade-Off zwischen Länge des prompts - Zeit - Qualität untersuchen
    # dynamische Promptlänge, abhängig von der Länge des errors
    n_pairs = min(int(300 / len(error)), len(error_correction_pairs))

    for (error, correction) in random.sample(error_correction_pairs, n_pairs):
        prompt = prompt + f"error:{error}" + '\n' + f"correction:{correction}" + '\n'
    prompt = prompt + f"error:{old_value}" + '\n' + "correction:"

    return prompt

def llm_master_prompt(cell: Tuple[int, int], df_error_free_subset: pd.DataFrame, df_row_with_error: pd.DataFrame) -> str:
    """
    Generate the llm_master prompt sent to the LLM.
    """

    prompt = "You are a data cleaning machine that returns a correction, which is a single expression. If "\
                "you do not find a correction, return the token <NULL>. You always follow the example.\n---\n"
    n_pairs = min(5, len(df_error_free_subset))
    rows = random.sample(range(len(df_error_free_subset)), n_pairs)
    for row in rows:
        row_as_string, correction = error_free_row_to_prompt(df_error_free_subset, row, cell[1])
        prompt = prompt + row_as_string + '\n' + f'correction:{correction}' + '\n'
    final_row_as_string, _ = error_free_row_to_prompt(df_row_with_error, 0, cell[1])
    prompt = prompt + final_row_as_string + '\n' + 'correction:'

    return prompt