from datasets import Dataset
from datasets import load_from_disk, concatenate_datasets
import json
import yaml
import os

desired_dir = "/llm-project/src/"
os.chdir(desired_dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)
DATASET_PATH = os.path.join(SCRIPT_DIR, config['dataset']['dataset_dev_path'])
TABLES_PATH = os.path.join(SCRIPT_DIR, config['dataset']['tables_dev_path'])
SYSTEM_PROMPT = os.path.join(SCRIPT_DIR, config['system_prompt'])


def preprocess(dataset, tables, use_evidence):

    with open(dataset, "r") as f:
        data = json.load(f)

    with open(tables, "r") as f:
        schemas = {schema["db_id"]: schema for schema in json.load(f)}

    formatted_data = []
    for sample in data:
        db_id = sample["db_id"]
        schema = schemas.get(db_id, {})

        tables = schema.get("table_names", [])
        columns = [f"{col[1]} ({schema['column_types'][i]})" for i, col in enumerate(schema.get("column_names", []))]
        #schema_info = f"Tables: {', '.join(tables)}\nColumns: {', '.join(columns)}"
        schema_info = generate_ddl_from_json(TABLES_PATH, db_id)
        prompt = f"[SCHEMA] {schema_info} "

        if use_evidence:
            prompt += f"{sample['evidence'] }"

        prompt += "[/SCHEMA]"
        prompt += f"[QUESTION] {sample['question']} [/QUESTION]"

        '''
                prompt += """Given the above [SCHEMA] of a database and a question [QUESTION], translate the question into a valid SQLite statement.
            The final SQL statement should be enclosed between [SQL] and [/SQL]. Consider that the content you put between [SQL] and [/SQL] has to be the input of DBMS, so 
            don't explain your queries, don't use comments, and don't put any additional information that is not useful for query execution. In addition, before you write the final
            SQL query, explain in detail step by step how you would solve the problem, just as a human would do."""
        '''

        prompt += SYSTEM_PROMPT
        response = sample["SQL"]
        formatted_data.append({"problem": prompt, "answer": response, "db_id": db_id})
    
    dataset = Dataset.from_list(formatted_data)
    if use_evidence:
        dataset.save_to_disk("./sql_dataset_evidence")
    else:
        dataset.save_to_disk("./sql_dataset_no_evidence")
    return dataset



def concatenate_sql_datasets():
    dataset_with_evidence = load_from_disk("./sql_dataset_evidence")
    dataset_no_evidence = load_from_disk("./sql_dataset_no_evidence")
    concatenated_dataset = concatenate_datasets([dataset_with_evidence, dataset_no_evidence])
    concatenated_dataset.save_to_disk("./sql_dataset_combined")
    return concatenated_dataset



def generate_ddl_from_json(json_path, target_db_id):

    with open(json_path, 'r') as f:
        databases = json.load(f)
    
    db_data = next((db for db in databases if db['db_id'] == target_db_id), None)
    if not db_data:
        return f"Error: Database '{target_db_id}' not found"
    
    table_names = db_data['table_names'] 
    column_names = db_data['column_names']
    column_types = db_data['column_types']
    primary_keys = db_data['primary_keys']
    foreign_keys = db_data['foreign_keys']
    
    tables = {}
    for i, (table_idx, col_name) in enumerate(column_names):
        if table_idx != -1: 
            if table_idx not in tables:
                tables[table_idx] = []
            tables[table_idx].append({
                'name': col_name,
                'type': column_types[i]
            })
    
    pk_set = set()
    for pk in primary_keys:
        if isinstance(pk, list):
            pk_set.update(pk)
        else:
            pk_set.add(pk)
    
    fk_dict = {}
    for fk in foreign_keys:
        referencing_col = fk[0]
        referenced_col = fk[1]
        fk_dict[referencing_col] = referenced_col
    
    ddl_statements = []
    
    for table_idx, columns in tables.items():
        table_name = table_names[table_idx]
        ddl = f"CREATE TABLE \"{table_name}\" (\n"
        
        column_defs = []
        for i, col in enumerate(columns):
            col_def = f"    \"{col['name']}\" {col['type']}" 
            col_idx = column_names.index([table_idx, col['name']])
            
            if col_idx in pk_set:
                col_def += " PRIMARY KEY"
            
            column_defs.append(col_def)
        
        for i, col in enumerate(columns):
            col_idx = column_names.index([table_idx, col['name']])
            if col_idx in fk_dict:
                ref_col_idx = fk_dict[col_idx]
                ref_table_idx = next(c[0] for c in column_names if c[1] == column_names[ref_col_idx][1])
                ref_table = table_names[ref_table_idx]
                ref_col = column_names[ref_col_idx][1]
                fk_def = f"    FOREIGN KEY (\"{col['name']}\") REFERENCES \"{ref_table}\"(\"{ref_col}\")"
                column_defs.append(fk_def)
        
        ddl += ",\n".join(column_defs)
        ddl += "\n);"
        ddl_statements.append(ddl)
    
    return "\n\n".join(ddl_statements)







if __name__ == "__main__":
    preprocess(DATASET_PATH, TABLES_PATH, True)
    preprocess(DATASET_PATH, TABLES_PATH, False)
    combined_dataset = concatenate_sql_datasets()
    print(f"Total samples in combined dataset: {len(combined_dataset)}")