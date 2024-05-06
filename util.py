import logging
import json
import os
import pickle
from sentence_transformers import SentenceTransformer, util

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
    
def create_save_embedding_model(papers, model,pool, output_file) -> None:
    corpus = [paper['title'] + '[SEP]' + paper['abstract'] for paper in papers]

    logger.info(f"Corpus size: {len(corpus)}")
    # logger.info(corpus[0])
    try:
        corpus_embeddings = model.encode_multi_process(corpus, pool, chunk_size=50)
    except Exception as e:
        logger.error(f"Error in encoding the corpus: {e}")
        return None
    
    try:
        with open(output_file, 'wb') as f_out:
            pickle.dump(corpus_embeddings, f_out)
    except Exception as e:
        logger.error(f"Error in saving the embeddings model to a file: {e}")
        return None
    
    return None


def load_embeddings(embeddings_file) -> None:
    try:
        with open(embeddings_file, 'rb') as f_in:
            embeddings = pickle.load(f_in)
    except Exception as e:
        logger.error(f"Error in loading the embeddings model from a file: {e}")
        return None
    
    return embeddings



def search_papers(model, title, abstract, corpus_embeddings):
    query_embedding = model.encode(title +'[SEP]'+ abstract, convert_to_tensor=True)

    search_hits = util.semantic_search(query_embedding, corpus_embeddings)
    search_hits = search_hits[0]  #Get the hits for the first query

    logger.info("Paper:", title)
    logger.info("Most similar papers:")
    results = []
    for hit in search_hits:
        related_paper = papers[hit['corpus_id']]
        logger.info("{:.2f}\t{}\t{} {}".format(hit['score'], related_paper['title'], related_paper['venue'], related_paper['year']))
        results.append(related_paper)
        
    return results

if __name__ == "__main__":

    input_file_path = "arxiv-metadata-oai-snapshot.json"
    output_file_path = "filtered_arxiv.json"
    keys_to_keep = ["title", "abstract", "categories", "authors", "journal-ref"]

    if not os.path.exists(output_file_path):
        filter_json(input_file_path, output_file_path, keys_to_keep)
    
    papers = []
    with open(output_file_path) as fIn:
        papers = json.load(fIn)
        print(type(papers))

    logger.info(f"{len(papers)} papers loaded")
    
    model_name = 'allenai-specter'
    model_path = 'cache_models_v2'
    model = None
    if not os.path.exists('cache_models_v2'):
        os.makedirs('cache_models_v2')
        model = SentenceTransformer(model_name)
        model.save(model_path)
    else:
        model = SentenceTransformer(model_path)

    pool = model.start_multi_process_pool(['cpu', 'cpu', 'cpu', 'cpu'])
    embeddings_file = 'corpus_embeddings.pkl'

    create_save_embedding_model(papers, model,pool, embeddings_file)
        
    # Now the embeddings are saved in the file corpus_embeddings.pkl and we have to load it for inference
    corpus_embeddings = load_embeddings(embeddings_file) # check the time to load for scope of optimization later
    # After loading the embeddings, we can search for papers
    model.stop_multi_process_pool(pool)
    results = search_papers(model, title="Deep Learning in Neural Networks: An Overview",
                            abstract = "In recent years, deep artificial neural networks (including recurrent ones) have won numerous contests in pattern recognition and machine learning. This historical survey compactly summarises relevant work, much of it from the previous millennium. Shallow and deep learners are distinguished by the depth of their credit assignment paths, which are chains of possibly learnable, causal links between actions and effects. I review deep supervised learning (also recapitulating the history of backpropagation), unsupervised learning, reinforcement learning & evolutionary computation, and indirect search for short programs encoding deep and large networks.", corpus_embeddings=corpus_embeddings)

