import os
import re
import yaml
import sqlite3
import sqlparse
import torch
import gc
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from google import genai
from google.genai import types
import random
from preprocess import generate_ddl_from_json



def decode_output(model, inputs, tokenizer, strategy):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)
    MAX_TOKENS = config['model']['max_tokens']
    N = config['inference']['n']

    input_length = inputs.input_ids.shape[1]
    
    match strategy.lower():
        case "greedy":
            outputs = [model.generate(**inputs,max_new_tokens=MAX_TOKENS,do_sample=False)]
            predictions = [tokenizer.decode(output[0, input_length:].tolist(), skip_special_tokens=True) for output in outputs]
        case "random_sampling":
            outputs = model.generate(**inputs,max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7,top_k=50, top_p=0.95, num_return_sequences=N)
            predictions = [tokenizer.decode(output[input_length:].tolist(), skip_special_tokens=True) for output in outputs]
        case "beam_search":
            outputs = model.generate(**inputs,max_new_tokens=MAX_TOKENS, num_beams=5,do_sample=False, early_stopping=True, num_return_sequences=N)
            predictions = [tokenizer.decode(output[input_length:].tolist(), skip_special_tokens=True) for output in outputs]
        case _:
            raise ValueError("Invalid decoding strategy. Choose 'greedy', 'random_sampling', or 'beam_search'")
    
    sqls = [extract_sql_query(sql) for sql in predictions]
    sqls = [normalize_sql(sql) for sql in sqls]
    return sqls



def test_time_scaling(sql_predictions, db_id, count_gemini_api, question_evidence, strategy):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)



    match strategy.lower():
        case "none":
            return sql_predictions[0], count_gemini_api
        
        


        case "best_of_n":
            scores = [execute_query(sql, db_id)["score"] for sql in sql_predictions]
            final_sql, max_score = max(zip(sql_predictions, scores), key=lambda x: x[1])
            return final_sql, count_gemini_api
        


        case "majority_voting":

            successful_sqls = []
            for sql in sql_predictions:
                result = execute_query(sql, db_id)
                success = result["connection_successful"]
                rows = result["data"]
                if success:
                    hashable_rows = tuple(tuple(row) if isinstance(row, list) else row for row in rows)
                    successful_sqls.append((sql, hashable_rows))
            
            if not successful_sqls:
                return sql_predictions[0], count_gemini_api
            
            rows_counts = Counter(pair[1] for pair in successful_sqls)
            if not rows_counts:
                return sql_predictions[0], count_gemini_api
                
            most_common_rows = rows_counts.most_common(1)[0][0]
            for sql, row in successful_sqls:
                if row == most_common_rows:
                    return sql, count_gemini_api
            return successful_sqls[0][0], count_gemini_api



        case "best_of_n_llm_judge":

            results = [execute_query(sql, db_id) for sql in sql_predictions]
            scores = [result['score'] for result in results]
            if all(score == 0 for score in scores):
                print(f"None of the queries were successful. Picking one random.")
                random_result = random.choice(results)
                return random_result['query'], count_gemini_api
            elif all(score == 0.5 for score in scores):
                print(f"All queries were valid but returned 0 rows. Picking one of them.")
                random_result = random.choice(results)
                return random_result['query'], count_gemini_api
            elif scores.count(0.5) >= 1 and scores.count(1) == 0:
                print(f"At least one query was valid but returned 0 rows. Picking that one.")
                candidate = next(result for result in results if result['score'] == 0.5)
                return candidate['query'], count_gemini_api
            elif scores.count(1) == 1:
                print(f"One query was successful. Picking that one.")
                candidate = next(result for result in results if result['score'] == 1)
                return candidate['query'], count_gemini_api
            elif scores.count(1) > 1:
                print(f"More queries were successful. LLM is judging which is the best one.")
                candidates = [result['query'] for result in results if result['score'] == 1]

                queryText = ""
                for i, query in enumerate(candidates, start=1):
                    queryText += f"{i}: {query}\n"

                ddl = generate_ddl_from_json(target_db_id=db_id)
                prompt =f"""
You are an expert SQL assistant tasked with choosing the best query among a list of queries. I will provide you with:
1. A list of SQL queries numbered.
2. User need.
3. The DDL (Data Definition Language) of the database, which describes the schema and structure of the tables involved.

Your job is to:
- Choose the best query, and put the number of the query you have choosen between the tag [CHOICE] [/CHOICE]. Example:

1: query_1
2: query_2

Final result: [CHOICE]2[/CHOICE].

Here are the details:

**SQL Queries:**
{queryText}

**User need:**
{question_evidence}

**Database DDL:**
{ddl}
"""

                response, count_gemini_api = generate_gemini(count_gemini_api, prompt=prompt)
                match = re.search(r'\[CHOICE\]\s*(\d+)\s*\[/CHOICE\]', response)
                number = 0
                if match:
                    number = int(match.group(1))
                    print(f"Extracted number: {number}")
                else:
                    print("No choice found")

                return candidates[number-1], count_gemini_api



        case "best_of_n_self_correcting":

            MAX_ITERATIONS = config['inference']['max_iterations']
            iteration = 0
            final_result = None

            while iteration < MAX_ITERATIONS and final_result is None:
                print(f"[Iteration {iteration + 1}]: ")
                results = [execute_query(sql, db_id) for sql in sql_predictions]
                scores = [result['score'] for result in results]
                
                if all(score == 0 for score in scores):
                    print(f"[Iteration {iteration+1}]: None of the queries were successful. Trying to adjust them...")
                    random_result = random.choice(results)
                    selected_query = random_result['query']
                    error_message = random_result['error_message']
                    ddl = generate_ddl_from_json(target_db_id=db_id)

                    prompt = f"""
You are an expert SQL assistant tasked with fixing a broken SQL query. I will provide you with:
1. A SQL query that is failing.
2. The error message generated when the query was executed.
3. The DDL (Data Definition Language) of the database, which describes the schema and structure of the tables involved.
4. User need.

Your job is to:
- Analyze the provided SQL query, the error message, and the database DDL.
- Identify the issue causing the error.
- Return a corrected version of the SQL query that resolves the error and aligns with the database schema.
- Enclose the corrected SQL query within [SQL] [/SQL] tags for easy extraction.

Here are the details:

**SQL Query:**
{selected_query}

**Error Message:**
{error_message}

**Database DDL:**
{ddl}

**User need:**
{question_evidence}
"""
                    sql_predictions, count_gemini_api = generate_gemini(count_gemini_api, prompt=prompt)
                    sql_predictions = [sql_predictions]
                    sql_predictions = [extract_sql_query(sql) for sql in sql_predictions]
                    sql_predictions = [extract_sql_markdown(sql) for sql in sql_predictions]
                
                elif 0.5 in scores:
                    print(f"[Iteration {iteration+1}]: One query was successful, but has returned 0 rows. Trying to adjust it...")
                    candidate = next(result for result in results if result['score'] == 0.5)
                    selected_query = candidate['query']
                    ddl = generate_ddl_from_json(target_db_id=db_id)
                    
                    prompt = f"""
You are an expert SQL assistant tasked with fixing a broken SQL query. I will provide you with:
1. A SQL query that returns 0 rows.
2. The DDL (Data Definition Language) of the database, which describes the schema and structure of the tables involved.
3. User need.

Your job is to:
- Analyze the provided SQL query and the database DDL.
- Identify the issue why it returns 0 rows.
- Return a corrected version of the SQL query that resolves the error and aligns with the database schema.
- Enclose the corrected SQL query within [SQL] [/SQL] tags for easy extraction.

Here are the details:

**SQL Query:**
{selected_query}

**Database DDL:**
{ddl}

**User need:**
{question_evidence}
"""
                    sql_predictions, count_gemini_api = generate_gemini(count_gemini_api, prompt=prompt)
                    sql_predictions = [sql_predictions]
                    sql_predictions = [extract_sql_query(sql) for sql in sql_predictions]
                    sql_predictions = [extract_sql_markdown(sql) for sql in sql_predictions]
                
                elif 1.0 in scores:
                    print(f"[Iteration {iteration+1}]: One query was successful and it return some rows. Checking if it is the correct one...")
                    successful_result = next(result for result in results if result['score'] == 1.0)

                    selected_query = successful_result['query']
                    data = successful_result['data']
                    ddl = generate_ddl_from_json(target_db_id=db_id)

                    prompt = f"""
You are an expert SQL assistant tasked with fixing a broken SQL query. I will provide you with:
1. A SQL query.
2. Results coming from the execution of this SQL query (only the first 5 rows).
3. The DDL (Data Definition Language) of the database, which describes the schema and structure of the tables involved.
4. User need.

Your job is to:
- Analyze the provided SQL query and the database DDL.
- Return a corrected version of the SQL query. If in your opinion the SQL query is right, don't change it.
- Enclose the corrected SQL query within [SQL] [/SQL] tags for easy extraction.

Here are the details:

**SQL Query:**
{selected_query}

**Data extracted (limited to 5 rows):**
{data}

**Database DDL:**
{ddl}

**User need:**
{question_evidence}
"""
                    sql_predictions, count_gemini_api = generate_gemini(count_gemini_api, prompt=prompt)
                    sql_predictions = [sql_predictions]
                    sql_predictions = [extract_sql_query(sql) for sql in sql_predictions]
                    sql_predictions = [extract_sql_markdown(sql) for sql in sql_predictions]
                    final_result = sql_predictions[0]
                
                iteration += 1

            if final_result != None:
                return final_result, count_gemini_api
            else:
                return sql_predictions[0], count_gemini_api


def execute_query(query, db_id):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)
    TABLES_DEV_PATH = config['dataset']['db_sqlite']
    db_path = os.path.join(TABLES_DEV_PATH, f"{db_id}", f"{db_id}.sqlite")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        num_rows = len(rows)
        conn.close()
        
        limited_rows = rows[:5]
        result = {"query": query, "connection_successful": True, "number_of_rows": num_rows, "data": [tuple(row) for row in limited_rows]} 
    
    except Exception as e:
        result = {"query": query, "connection_successful": False, "number_of_rows": 0, "data": [], "error_message": str(e)}
    
    score = 0
    if result["connection_successful"]:
        score += 1
    if result["data"] != []:
        score += 1
    
    result["score"] = score / 2
    return result



def normalize_sql(query):
    return ' '.join(query.split()).strip()



def extract_sql_query(sql_query):
    if not isinstance(sql_query, str):
        return None
        
    match = re.search(r'\[SQL\](.*?)\[/SQL\]', sql_query, re.DOTALL)
    return match.group(1) if match else sql_query



def extract_sql_markdown(sql_query):
    if not isinstance(sql_query, str):
        return None
        
    match = re.search(r'```sql\s*(.*?)\s*```', sql_query, re.DOTALL)
    return match.group(1) if match else sql_query



def generate_gemini(count_gemini_api, prompt=""):

    api_keys = [
        "AIzaSyAqRW_8EaZPd1UsMlFpxWWG_tZCuYwyIOg",
        'AIzaSyDqCkY8FCuTor7XoGWpN0Mo_zu2mUs4nlA',
        'AIzaSyCsNvVQI-ICAGOv8oTOC_tEFao4_ASVMr0',
        'AIzaSyBXN6aLMXxSRUF3A2QyDf_s_XQT0jHtvco',
        'AIzaSyAfsCfuno7TtcaWGQvMraweCfuQYj4tIZs',
        'AIzaSyCGAZXGZbReh3hu_baMCz9Cdc1EKHI6TBg',
        'AIzaSyBBoMpaJZMPSEpgTLIpUaaxpvOXV2QoFxE',
    ]

    current_api_key = api_keys[count_gemini_api % len(api_keys)]
    count_gemini_api = count_gemini_api + 1
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)
    GEMINI_KEY = current_api_key
    MODEL_NAME = config['inference']['model_name_validator']
    MAX_TOKENS = config['model']['max_tokens']

    client = genai.Client(api_key=GEMINI_KEY)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt),],),]
    generate_content_config = types.GenerateContentConfig(temperature=0, top_p=0.95, top_k=40, max_output_tokens=MAX_TOKENS, response_mime_type="text/plain",)

    response = ""
    for chunk in client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=generate_content_config,):
        response = response + chunk.text
    
    return response, count_gemini_api