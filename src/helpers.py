import itertools
import json
import random
import re
import sqlite3
import sys
import threading
import time

import numpy as np
from typing import Union, Dict, Tuple, List, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

import openai
import requests
import pandas as pd


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


class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.busy = False
        self.delay = 0.1
        self.message = message
        self.thread = None

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _spin(self):
        while self.busy:
            self.write(f"\r{self.message} {next(self.spinner)}")
            time.sleep(self.delay)
        self.write("\r\033[K")

    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self._spin())
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        time.sleep(self.delay)
        if self.thread:
            self.thread.join()
        self.write("\r")


def connect_to_cache() -> sqlite3.Connection:
    """
    Connect to the cache for LLM prompts.
    @return: a connection to the sqlite3 cache.
    """
    conn = sqlite3.connect('cache.db')
    return conn


def is_column_in_cache(dataset: str, column: int, correction_model_name: str, llm_name: str) -> bool:
    conn = connect_to_cache()
    cursor = conn.cursor()

    # Check if a record with the same values already exists
    cursor.execute(
        """SELECT COUNT(*)
           FROM cache
           WHERE dataset = ? AND column = ? AND
                 correction_model = ? AND llm_name = ?""",
        (dataset, column, correction_model_name, llm_name)
    )
    existing_records = cursor.fetchone()[0]

    # wenn kein Eintrag vorhanden ist (existing_records = 0) dann false ansonsten true
    return existing_records > 0


def fetch_cache(dataset: str,
                error_cell: Tuple[int, int],
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
    @param llm_name: Name of the llm.
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


def llm_test_user_corrections(llm_name: str, session: requests.Session, pairs: list[tuple[str, str]], column_name: str, category: str) -> float and list[tuple[str, str]]:
    llama_url = "http://localhost:8080/completion"
    valid_corrections: int = 0
    failed_pairs: list[tuple[str, str]] = []

    for (old_value, correction) in pairs:
        pairs_temp: list[tuple[str, str]] = pairs.copy()
        pairs_temp.remove((old_value, correction))
        prompt = llm_correction_prompt(old_value, pairs_temp, column_name, category)

        payload = {
            "model": llm_name,
            "prompt": prompt,
            "n_predict": len(old_value),
            "n_probs": len(old_value),
            "stream": False,
            "return_tokens": True
        }
        try:
            response = send_request_to_llama_server(llama_url=llama_url, payload=payload, timeout=60, max_retries=10,
                                                    session=session)

            response.raise_for_status()
            response_data = response.json()
            generated_text = response_data.get("content", "")
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0]
            if generated_text == correction:
                valid_corrections += 1
            else:
                failed_pairs.append((old_value, correction))
        except requests.RequestException as e:
            print(f"Fehler bei der Anfrage an llama.cpp-Server: {e}")
            failed_pairs.append((old_value, correction))

    correction_quota: float = valid_corrections / len(pairs)
    return correction_quota, failed_pairs


def generate_llm_response(
        prompt: Union[str, None],
        response_length: int,
        dataset: str,
        error_cell: Tuple[int, int],
        correction_model_name: str,
        session: requests.Session,
        error_fraction: Union[None, int] = None,
        version: Union[None, int] = None,
        error_class: Union[None, str] = None,
        llm_name: str = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
) -> Union[LLMResult, None]:
    """
    Generiert eine Antwort mit llama.cpp über den REST-Server.
    """

    # Falls kein Prompt vorhanden ist, macht eine Anfrage keinen Sinn
    if not prompt:
        return None

    llama_url: str = "http://localhost:8080/completion"

    payload: dict[str, Any] = {
        "model": llm_name,
        "prompt": prompt,
        "n_predict": response_length,
        "n_probs": response_length,
        "stream": False,
        "return_tokens": True,
    }

    try:
        response = send_request_to_llama_server(llama_url=llama_url, payload=payload, timeout=60, max_retries=10,
                                                session=session)
        response.raise_for_status()

        response_data: dict[str, Any] = response.json()
        completion_probs: list[dict[str, Any]] = response_data.get("completion_probabilities", [])

        # Sammle alle Token-Informationen bis zum ersten Token, das "\n" enthält.
        filtered_tokens_info: list[dict[str, Any]] = []
        for item in completion_probs:
            token: str = item.get("token", "")
            if "\n" in token:
                # Extrahiere den Teil vor dem "\n"
                trimmed: str = token.split("\n")[0]
                if trimmed:  # Nur hinzufügen, wenn der extrahierte Teil nicht leer ist
                    new_item: dict[str, Any] = item.copy()
                    new_item["token"] = trimmed
                    filtered_tokens_info.append(new_item)
                break  # Beende die Schleife, sobald ein Token mit "\n" gefunden wurde
            else:
                filtered_tokens_info.append(item)

        # Erstelle Listen für die Korrektur-Tokens, logprobs und Top-Logprobs.
        correction_tokens: list[Any] = [item.get("token") for item in filtered_tokens_info]
        token_logprobs: list[Any] = [item.get("logprob") for item in filtered_tokens_info]
        top_logprobs: list[Any] = []
        """top_logprobs = [
            {p.get("token"): p.get("logprob") for p in item.get("top_logprobs", [])}
            for item in filtered_tokens_info
        ]"""

    except requests.RequestException as e:
        print(f"Fehler bei der Anfrage an llama.cpp-Server: {e}")
        return None

    row, column = error_cell
    llm_result = LLMResult(dataset, row, column, correction_model_name, correction_tokens, token_logprobs, top_logprobs,
                           error_fraction, version, error_class, llm_name)
    return llm_result


def send_request_to_llama_server(
        llama_url: str,
        payload: dict[str, Any],
        timeout: Union[int, float],
        max_retries: int,
        session: requests.Session
        ) -> requests.Response:
    """
    Sendet ein Prompt-Payload an llama.cpp und versucht bei 503 (Loading model)
    mehrmals, die Anfrage erneut abzuschicken.
    """

    for attempt in range(max_retries):
        response = session.post(llama_url, json=payload, timeout=timeout)

        if response.status_code != 503:
            return response

        # Falls Status 503, schauen wir ins JSON
        try:
            data: dict[str, Any] = response.json()
        except ValueError:
            # Falls kein gültiges JSON zurückkommt, brechen wir ab
            return response

        error: dict[str, Any] = data.get("error", {})

        # Prüfen, ob "Loading model" als Grund angegeben ist
        if (error.get("code") == 503 and
                error.get("message") == "Loading model"):
            print(f"Server meldet: {error.get('message')}. "
                  f"Versuch {attempt + 1} von {max_retries}, warte 3s ...")
            time.sleep(3)
            # Dann erneut versuchen
        else:
            # Irgendein anderer 503-Fehler
            return response

    # Nach max_retries wird die letzte response zurückgegeben
    return response


def send_request_to_lmstudio_server(model: str, prompt: str, request_timeout: int) -> str:
    openai.api_key = ""
    openai.api_base = "http://127.0.0.1:1234/v1"
    response: str = ""

    for attempt in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                top_k=15,
                request_timeout=request_timeout
            )
            return response
        except Exception as e:
            # print(e)
            # return ""
            time.sleep(3)

    return response


def parse_response(response: str, keyword: str) -> str:
    content = response["choices"][0]["message"]["content"]

    if keyword in content:
        content = content.split(keyword, 1)[1].lstrip("\n")

    # Markdown entfernen
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    if content.endswith("```"):
        lines = content.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    if "```" in content:
        content = content.split("```", 1)[0].lstrip("\n")

    return content


def llm_recognize_correction(column_name, column_category, error_correction_pairs):
    """
    Diese Methode orchestriert den Self-Revision-Loop über die OpenAI API:
      - Er erstellt einen initialen Prompt anhand von column_name, column_category und error_correction_pairs.
      - Nach jeder Antwort wird geprüft, ob der confidence-Wert den gewünschten Threshold erreicht.
      - Falls nicht, wird eine Revision angefordert, indem das bisherige Ergebnis in den Prompt aufgenommen wird.

    Rückgabe: JSON-Antwort des LLM mit der Transformation, einer Erklärung und dem confidence-Wert.
    """

    prompt: str = llm_recognize_correction_prompt(error_correction_pairs)

    # with Spinner("Thinking..."):
    response: str = send_request_to_lmstudio_server(model="deepseek-r1-distill-qwen-14b", prompt=prompt, request_timeout=1320)

    if response is "":
        return ""
    else:
        return parse_response(response, "</think")


def set_local_context(content: str) -> Union[Callable[..., Any], dict]:
    code: str = content
    lines: list = content.splitlines()
    match: re.Match[str] | None = re.search(r"def\s+(\w+)\s*\(", lines[0])
    method_name: str = ""

    if match:
        method_name = match.group(1)

    if not code or method_name is "":
        # raise ValueError("Die generierte Antwort enthält nicht alle erforderlichen Felder.")
        print("Die generierte Antwort enthält nicht alle erforderlichen Felder.")
        return {}

    # Führe den generierten Code in einem separaten lokalen Kontext aus
    local_context: dict = {}
    try:
        exec(code, globals(), local_context)
    except Exception as e:
        # raise RuntimeError(f"Fehler beim Ausführen des generierten Codes: {e}")
        print(e)
        return {}

    # Suche die generierte Methode im lokalen Kontext
    if method_name not in local_context:
        # raise ValueError(f"Die Methode '{method_name}' wurde im generierten Code nicht gefunden.")
        print("Die Methode '{method_name}' wurde im generierten Code nicht gefunden.")
        return {}

    return local_context[method_name]


def test_generated_method(generated_method, error_correction_pairs: list[tuple[str, str]]) -> float and dict:
    passed_tests: int = 0
    failed_test: dict = {}

    for error_val, expected_correction in error_correction_pairs:
        try:
            result = generated_method(error_val)
            output = str(result)
        except Exception as e:
            output = f"Error: {e}"
        # Vergleiche den Output mit der erwarteten Korrektur
        if output == expected_correction:
            passed_tests += 1
        else:
            # print(f"Test fehlgeschlagen für '{error_val}': erwartet '{expected_correction}', erhalten '{output}'")
            failed_test.__setitem__(error_val, (expected_correction, output))

    # print(f"Anzahl bestandener Tests: {passed_tests} von {len(error_correction_pairs)}")
    quota = passed_tests / len(error_correction_pairs)

    return quota, failed_test


def llm_generate_method(initial_response, error_correction_pairs) -> str and float and dict and Callable[..., Any]:
    """
    Diese Methode generiert aus der Transformation den Python-Code:
      - Sie erstellt einen Prompt, der den vom Reasoning-Modell erstellten Transformation-Prozess (procedure) nutzt.
      - Sie ruft das LLM auf, um eine Methode zu generieren, die Fehler systematisch korrigiert.
      - Sie führt den generierten Code aus und testet die Methode anhand der gegebenen Error-Correction-Pairs.

    Rückgabe: JSON-Antwort des LLM mit dem Methodennamen, dem Code und der Anzahl der bestandenen Unittests.
    """
    prompt: str = llm_generate_method_prompt(initial_response, error_correction_pairs)
    # with Spinner("Generating..."):
    response: str = send_request_to_lmstudio_server(model="codestral-22b-v0.1", prompt=prompt, request_timeout=240)

    if response is "":
        return "", 0, {}, None

    content: str = parse_response(response, keyword="python")
    generated_method: Callable[..., Any] = set_local_context(content)

    if generated_method == {}:
        return "", 0, {}, None

    quota, failed_tests = test_generated_method(generated_method, error_correction_pairs)

    return content, quota, failed_tests, generated_method


def llm_recognize_method_error(transformation: str, code: str, failed_tests: dict) -> str:

    prompt: str = llm_recognize_method_error_prompt(transformation, code, failed_tests)
    response: str = send_request_to_lmstudio_server(model="deepseek-r1-distill-qwen-14b", prompt=prompt, request_timeout=900)

    if response is "":
        return ""

    content: str = parse_response(response, "</think>")

    return content


def llm_generate_improved_method_failed_tests(transformation: str, failed_tests: dict, code: str, error_correction_pairs: list[tuple[str, str]]) -> str and float and dict and Callable[..., Any]:

    prompt: str = llm_generate_improved_method_failed_tests_prompt(transformation, failed_tests, code)
    response: str = send_request_to_lmstudio_server(model="codestral-22b-v0.1", prompt=prompt, request_timeout=240)

    if response is "":
        return "", 0, {}, None

    content: str = parse_response(response, "python")
    generated_method: Callable[..., Any] = set_local_context(content)
    quota, failed_test = test_generated_method(generated_method, error_correction_pairs)

    return content, quota, failed_test, generated_method


def llm_generate_improved_method_revision(transformation: str, response: str, code: str, error_correction_pairs: list[tuple[str, str]]) -> str and float and dict and Callable[..., Any]:

    prompt: str = llm_generate_improved_method_revision_prompt(transformation, response, code)
    response: str = send_request_to_lmstudio_server(model="codestral-22b-v0.1", prompt=prompt, request_timeout=240)

    if response is "":
        return "", 0, {}, None

    content: str = parse_response(response, "python")
    generated_method: Callable[..., Any] = set_local_context(content)
    quota, failed_test = test_generated_method(generated_method, error_correction_pairs)

    return content, quota, failed_test, generated_method


def prepare_correction_for_cache(dataset: str, error_cell: Tuple[int, int], correction_model: str,
                                 correction: str, error_fraction: Union[int, None],
                                 version: Union[int, None], error_class: Union[str, None], llm_name: str) -> LLMResult:

    # die Method überführt die correction als ein einzelnes token mit einem logprob von 0, top_logprobs wird leer gelassen
    return LLMResult(dataset, error_cell[0], error_cell[1], correction_model, [correction], [0], [], error_fraction,
                     version, error_class, llm_name)


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
        (llm_result.dataset, llm_result.row, llm_result.column, llm_result.correction_model_name,
         llm_result.error_fraction, llm_result.version, llm_result.error_class, llm_result.llm_name)
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
              f"possible errors for this category: {map_error_types(category)}\n---\n"
              f"examples:")

    """ "neuer" prompt ist schlechter als der "alte" - für später als Referenz aufheben
    prompt = ("You are a data cleaning machine designed to detect patterns and return corrections. If no correction is "   
              "found, respond with the token <NULL>.  Always adhere to the provided format and return ONLY the correction "
              "or <NULL>.")
    """
    error, correction = error_correction_pairs[0]

    # dynamische Promptlänge, abhängig von der Länge des errors
    n_pairs = min(int(333 / len(error)), len(error_correction_pairs))

    for (error, correction) in random.sample(error_correction_pairs, n_pairs):
        prompt = prompt + f"error:{error}" + '\n' + f"correction:{correction}" + '\n'
    prompt = prompt + f"error:{old_value}" + '\n' + "correction:"

    return prompt


def llm_master_prompt(cell: Tuple[int, int], df_error_free_subset: pd.DataFrame,
                      df_row_with_error: pd.DataFrame) -> str and int:
    """
    Generate the llm_master prompt sent to the LLM.
    """
    prompt = "You are a data cleaning machine that returns a correction, which is a single expression. If " \
             "you do not find a correction, return the token <NULL>. You always follow the example.\n---\n"
    n_pairs = min(5, len(df_error_free_subset))
    rows = random.sample(range(len(df_error_free_subset)), n_pairs)
    max_length: int = 0
    for row in rows:
        row_as_string, correction = error_free_row_to_prompt(df_error_free_subset, row, cell[1])
        if len(correction) > max_length: max_length = len(correction)
        prompt = prompt + row_as_string + '\n' + f'correction:{correction}' + '\n'
    final_row_as_string, _ = error_free_row_to_prompt(df_row_with_error, 0, cell[1])
    prompt = prompt + final_row_as_string + '\n' + 'correction:'

    return prompt, max_length


def llm_recognize_correction_prompt(error_correction_pairs: List[Tuple[str, str]]) -> str:
    """
    Generate the prompt used to recognize the transformation from erroneous value to correction.
    """

    n_pairs = min(10, len(error_correction_pairs))
    error_correction_prompt = ""
    for (error, correction) in random.sample(error_correction_pairs, n_pairs):
        error_correction_prompt = error_correction_prompt + f"[\"{error}\" , \"{correction}\"], \n"

    prompt = f"""
    I have a very important submission tomorrow and a colleague deliberately made a mistake when entering the data. I have the following values as an example, where value1 is the wrong value that she entered and value2 is the same value corrected. The two values are the same and only differ in the transformation. You are a highly intelligent error correction machine, which is why I am giving you this task of the utmost importance. My colleague has entered all values incorrectly with the same (illogical) transformation. We now have to find transformation from value1 to value2!
    
    Error-Correction Pairs: 
    {error_correction_prompt}
    """

    return prompt


def llm_generate_method_prompt(transformation, error_correction_pairs: List[Tuple[str, str]]) -> str:
    """
    Generate the method-generation prompt sent to the LLM.
    """

    n_pairs = min(5, len(error_correction_pairs))
    error_correction_prompt = ""
    for (error, correction) in random.sample(error_correction_pairs, n_pairs):
        error_correction_prompt = error_correction_prompt + f"[\"{error}\" , \"{correction}\"], \n"

    prompt = f"""
    You are an expert software developer. You need to write an python method that transforms a given String, using the following transformation. You respond with nothing but the code.

    The transformation that should be represented by the method:
    {transformation}
    """

    return prompt


def llm_recognize_method_error_prompt(transformation: str, code: str, failed_tests: dict) -> str:
    """

    :param transformation:
    :param code:
    :param failed_tests:
    :return:
    """

    prompt: str = f"""
        You are an expert software developer. I have developed a python method for the following transformation but the method doesnt work in {len(failed_tests)} test. You need to help me analyze what is missing in the method.
        
        Here is what I want to do with the method:
        {transformation}
        
        And here is the method I wrote:
        {code}
        
        But it failed in these testcases:
    """

    for error_val, (expected, provided) in failed_tests.items():
        prompt += f"\nTest failed for: '{error_val}': expected '{expected}', method output '{provided}'"

    return prompt


def llm_generate_improved_method_revision_prompt(transformation: str, response: str, code: str) -> str:
    prompt = f"""
    You are an expert software developer. I have developed a python method for the transformation but the method doesnt work. You need to correct my method.
    
    Here is what I want to do with the method:
    {transformation}
    
    And here is the method I wrote:
        {code}
        
    And this is what ChatGPT said is missing in my method:
    {response} 
    """

    return prompt


def llm_generate_improved_method_failed_tests_prompt(transformation: str, failed_tests: dict, code: str) -> str:
    prompt = f"""
    You are an expert software developer. I have developed a python method for the transformation but the method doesnt work. You need to correct my method.

    Here is what I want to do with the method:
    {transformation}

    And here is the method I wrote:
        {code}

    But it failed in these testcases:
    """

    for error_val, (expected, provided) in failed_tests.items():
        prompt += f"\nTest failed for: '{error_val}': expected '{expected}', method output '{provided}'"

    return prompt
