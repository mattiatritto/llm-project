import os
import re
import yaml
import sqlite3
import sqlparse
import torch
from collections import Counter

def decode_output(model, inputs, db_id, tokenizer, strategy="greedy"):

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    desired_dir = "/llm-project/src/"
    os.chdir(desired_dir)
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")

    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)

    
    MAX_TOKENS = config['model']['max_tokens']
    N = config['inference']['n']
    TABLES_DEV_PATH = config['dataset']['db_sqlite']

    input_length = inputs.input_ids.shape[1]
    
    match strategy.lower():
        case "greedy":
            output = model.generate(**inputs,max_new_tokens=MAX_TOKENS,do_sample=False)
        case "random_sampling":
            output = model.generate(**inputs,max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7,top_k=50, top_p=0.95)
        case "beam_search":
            output = model.generate(**inputs,max_new_tokens=MAX_TOKENS, num_beams=5,do_sample=False, early_stopping=True)
        case "best_of_n":
            outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=N,return_dict_in_generate=True,output_scores=True)
            decoded_outputs = [tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in outputs.sequences]
            
            db_path = os.path.join(TABLES_DEV_PATH, f"{db_id}", f"{db_id}.sqlite")
            results = []
            for i, output in enumerate(decoded_outputs):
                breakpoint()
                output = extract_sql_query(output)
                output = normalize_sql(output)
                if not is_valid_sql(output):
                    results.append((output, False, False, 0.0))
                    continue
                
                success, has_rows = execute_query(output, db_path)
                

                transition_scores = outputs.scores
                seq_score = sum(torch.log_softmax(scores, dim=-1).max(dim=-1).values for scores in transition_scores) / len(transition_scores)
                score = seq_score.max().item() if success else -float('inf')
                
                results.append((output, success, has_rows, score))
            
            # Pick the best output:
            # 1. First query that executes successfully and returns rows
            # 2. If none return rows, pick the highest-scoring successful query
            # 3. If none succeed, pick the highest-scoring overall

            best_output = decoded_outputs[0]  # Fallback
            best_score = -float('inf')
            found_rows = False
            
            for output, success, has_rows, score in results:
                if success and has_rows:
                    return output  # Return the first query with rows
                elif success and score > best_score:
                    best_score = score
                    best_output = output
            
            return best_output
        
        case "majority_voting":
            outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=N, return_dict_in_generate=True, output_scores=True)
            decoded_outputs = [tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in outputs.sequences]
            db_path = os.path.join(TABLES_DEV_PATH, f"{db_id}", f"{db_id}.sqlite")
            
            results = []
            for i, output in enumerate(decoded_outputs):
                output = extract_sql_query(output)
                norm_output = normalize_sql(output)
                if not is_valid_sql(norm_output):
                    results.append((norm_output, False, False, 0.0))
                    continue
                
                success, has_rows = execute_query(norm_output, db_path)
                
                transition_scores = outputs.scores
                seq_score = sum(torch.log_softmax(scores, dim=-1).max(dim=-1).values for scores in transition_scores) / len(transition_scores)
                score = seq_score.max().item() if success else -float('inf')
                results.append((norm_output, success, has_rows, score))
            
            vote_counter = Counter([output for output, success, _, _ in results if success])  # Count only successful executions
            if not vote_counter:
                # If no successful executions, fall back to the first output
                return decoded_outputs[0]
            

            # Find candidates with the highest vote count
            max_votes = max(vote_counter.values())
            top_candidates = [(output, count) for output, count in vote_counter.items() if count == max_votes]
            
            # Among top candidates, prioritize ones that return rows, then use score
            best_output = top_candidates[0][0]  # Default to first max-voted output
            best_score = -float('inf')
            
            for candidate, _ in top_candidates:
                for output, success, has_rows, score in results:
                    if output == candidate:
                        if has_rows:
                            return output  # Return first candidate with rows
                        elif score > best_score:
                            best_score = score
                            best_output = output
            
            return best_output
        case _:
            raise ValueError("Invalid strategy. Choose 'greedy', 'random', or 'beam'")
            
    return extract_sql_query(tokenizer.decode(output[0, input_length:].tolist(), skip_special_tokens=True))



def execute_query(query, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return True, len(rows) > 0
    except Exception as e:
        return False, False



def is_valid_sql(query):
    try:
        sqlparse.parse(query)
        return True
    except:
        return False



def normalize_sql(query):
    return ' '.join(query.split()).strip()



def extract_sql_query(sql_query):
    if not isinstance(sql_query, str):
        return None
        
    match = re.search(r'\[SQL\](.*?)\[/SQL\]', sql_query, re.DOTALL)
    return match.group(1) if match else sql_query
