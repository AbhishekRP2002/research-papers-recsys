import logging
import json
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")
logger.setLevel(logging.INFO)


def filter_json(input_file, output_file, keys_to_keep) -> None:
    filtered_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                json_obj = json.loads(line)
                filtered_obj = {key: json_obj[key] for key in keys_to_keep if key in json_obj}
                filtered_data.append(filtered_obj)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None

    try:
        with open(output_file, 'w') as f_out:
            json.dump(filtered_data, f_out, indent=4)
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return None
    
def fetch_papers_by_semantic_search():
    raise NotImplementedError("Not implemented yet")

if __name__ == "__main__":

    input_file_path = "arxiv-metadata-oai-snapshot.json"
    output_file_path = "filtered_arxiv.json"
    keys_to_keep = ["title", "abstract", "categories", "authors", "journal-ref"]

    if not os.path.exists(output_file_path):
        filter_json(input_file_path, output_file_path, keys_to_keep)
    
    with open(output_file_path) as fIn:
        papers = json.load(fIn)
        print(type(papers))

    logger.info(f"{len(papers)} papers loaded")
