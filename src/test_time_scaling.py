import os
import re
import yaml
import sqlite3
import sqlparse
import torch
from collections import Counter



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



def test_time_scaling(sql_predictions, db_id, strategy):

    match strategy.lower():
        case "none":
            return sql_predictions[0]

        case "best_of_n":
            scores = [score(sql, db_id) for sql in sql_predictions]
            final_sql, max_score = max(zip(sql_predictions, scores), key=lambda x: x[1])
            return final_sql
        
        case "majority_voting":
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            desired_dir = "/llm-project/src/"
            os.chdir(desired_dir)
            CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
            with open(CONFIG_PATH, 'r') as config_file:
                config = yaml.safe_load(config_file)
            TABLES_DEV_PATH = config['dataset']['db_sqlite']
            db_path = os.path.join(TABLES_DEV_PATH, f"{db_id}", f"{db_id}.sqlite")

            successful_sqls = []
            for sql in sql_predictions:
                success, rows = execute_query(sql, db_path)
                if success:
                    hashable_rows = tuple(tuple(row) if isinstance(row, list) else row for row in rows)
                    successful_sqls.append((sql, hashable_rows))
            
            if not successful_sqls:  # Handle case where no queries succeed
                return sql_predictions[0]  # Return first prediction as fallback
            
            rows_counts = Counter(pair[1] for pair in successful_sqls)
            if not rows_counts:  # Handle empty counter
                return sql_predictions[0]  # Return first prediction as fallback
                
            most_common_rows = rows_counts.most_common(1)[0][0]
            for sql, row in successful_sqls:
                if row == most_common_rows:
                    return sql
            return successful_sqls[0][0]  # Fallback if no match found
    


def execute_query(query, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, None



def normalize_sql(query):
    return ' '.join(query.split()).strip()



def extract_sql_query(sql_query):
    if not isinstance(sql_query, str):
        return None
        
    match = re.search(r'\[SQL\](.*?)\[/SQL\]', sql_query, re.DOTALL)
    return match.group(1) if match else sql_query



def score(query, db_id):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)
    TABLES_DEV_PATH = config['dataset']['db_sqlite']
    db_path = os.path.join(TABLES_DEV_PATH, f"{db_id}", f"{db_id}.sqlite")
    score = 0.0
    max_points = 2.0

    success, rows = execute_query(query, db_path)
    if success:
        score = score + 1
    if rows != None:
        score = score + 1

    return score / max_points