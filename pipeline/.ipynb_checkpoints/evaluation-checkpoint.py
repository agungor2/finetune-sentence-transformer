import logging
import argparse
import os
import tarfile
import json
import pandas as pd
import pathlib
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available. Using CPU.")
    return device


def shap_ire_results(df, prefix):
    ire_response = dict()
    for key, value in df.to_dict("records")[0].items():
        print(key)
        if (key != "epoch") and (key != "steps"):
            ire_response[f"{prefix}-{key}"] = {
                "value": value,
                "standard_deviation": "NaN"
            }
    return ire_response


def evaluate_top_hit(dataset, embeddings, top_k=5):
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']

    docs = [Document(metadata=dict(id_=id_), page_content=text) for id_, text in corpus.items()] 

    db = FAISS.from_documents(docs, embeddings)

    eval_results = []
    for query_id, query in queries.items():
        retrieved_docs = db.similarity_search(query, top_k)
        retrieved_ids = [doc.metadata['id_'] for doc in retrieved_docs]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            'is_hit': is_hit,
            'retrieved': retrieved_ids,
            'expected': expected_id,
            'query': query_id,
        }
        eval_results.append(eval_result)

    return eval_results


def evaluate_sentence_transformers(
    dataset,
    model,
    output_path,
    name,
):
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    return evaluator(model, output_path=output_path)


def load_sentence_transformer_with_fallback(model_name_or_path, device):
    """Load SentenceTransformer with device fallback."""
    try:
        logger.info(f"Attempting to load model '{model_name_or_path}' on device '{device}'")
        model = SentenceTransformer(model_name_or_path, device=device)
        logger.info(f"Successfully loaded model on {device}")
        return model
    except Exception as e:
        if device == "cuda":
            logger.warning(f"Failed to load model on CUDA: {e}")
            logger.info("Falling back to CPU...")
            try:
                model = SentenceTransformer(model_name_or_path, device="cpu")
                logger.info("Successfully loaded model on CPU")
                return model
            except Exception as cpu_e:
                logger.error(f"Failed to load model on CPU as well: {cpu_e}")
                raise cpu_e
        else:
            logger.error(f"Failed to load model: {e}")
            raise e


def load_huggingface_embeddings_with_fallback(model_name_or_path):
    """Load HuggingFaceEmbeddings with better error handling."""
    try:
        # Determine device for HuggingFaceEmbeddings
        device = get_device()
        model_kwargs = {'device': device}
        
        logger.info(f"Loading HuggingFaceEmbeddings for '{model_name_or_path}' on {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs=model_kwargs
        )
        logger.info("Successfully loaded HuggingFaceEmbeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load HuggingFaceEmbeddings: {e}")
        # Fallback to CPU if GPU fails
        if 'cuda' in str(e).lower():
            logger.info("Attempting fallback to CPU...")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name_or_path,
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("Successfully loaded HuggingFaceEmbeddings on CPU")
                return embeddings
            except Exception as cpu_e:
                logger.error(f"CPU fallback also failed: {cpu_e}")
                raise cpu_e
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id",
                        type=str,
                        default="sentence-transformers/msmarco-bert-base-dot-v5")
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data-path",
                        type=str,
                        default="/opt/ml/processing/input/data")
    parser.add_argument("--test-file",
                        type=str,
                        default="val_dataset.json")

    args, _ = parser.parse_known_args()

    # Detect available device
    device = get_device()

    # Load test data
    logger.debug("Load test data...")
    test_data = os.path.join(args.test_data_path, args.test_file)

    with open(test_data, 'r+') as f:
        test_dataset = json.load(f)

    # Load base model with fallback
    logger.debug("Load base model...")
    try:
        base_embeddings = load_huggingface_embeddings_with_fallback(args.base_model_id)
        
        eval_results = evaluate_top_hit(test_dataset, base_embeddings)
        df_base = pd.DataFrame(eval_results)
        base_top_hits = df_base['is_hit'].mean()
        logger.info(f"base model top hits: {base_top_hits}")
    except Exception as e:
        logger.error(f"Failed to evaluate base model: {e}")
        base_top_hits = 0.0  # Default value if evaluation fails

    # Load fine-tuned model
    logger.debug("Extracting the model...")
    model_file = os.path.join(args.model_path, args.model_file)
    
    if os.path.exists(model_file):
        file = tarfile.open(model_file)
        file.extractall(args.model_path)
        file.close()
    else:
        logger.warning(f"Model file {model_file} not found, assuming model is already extracted")

    logger.debug("Load fine tuned model...")
    try:
        finetuned_embeddings = load_huggingface_embeddings_with_fallback(args.model_path)
        
        eval_results = evaluate_top_hit(test_dataset, finetuned_embeddings)
        df_finetuned = pd.DataFrame(eval_results)
        finetuned_top_hits = df_finetuned['is_hit'].mean()
        logger.info(f"finetuned model top hits: {finetuned_top_hits}")
    except Exception as e:
        logger.error(f"Failed to evaluate finetuned model: {e}")
        finetuned_top_hits = 0.0  # Default value if evaluation fails

    # Sentence Transformers evaluation
    logger.info("Evaluate using InformationRetrievalEvaluator from sentence_transformers...")
    
    try:
        base_model = load_sentence_transformer_with_fallback(args.base_model_id, device)
        finetuned_model = load_sentence_transformer_with_fallback(args.model_path, device)

        tmp_path = "/tmp/results"
        pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)

        # Evaluate both models
        evaluate_sentence_transformers(test_dataset,
                                     base_model,
                                     output_path=tmp_path,
                                     name='base')

        evaluate_sentence_transformers(test_dataset,
                                     finetuned_model,
                                     output_path=tmp_path,
                                     name='finetuned')

        # Read results
        df_st_base = pd.read_csv(f"{tmp_path}/Information-Retrieval_evaluation_base_results.csv")
        df_st_finetuned = pd.read_csv(f"{tmp_path}/Information-Retrieval_evaluation_finetuned_results.csv")

        base_ire = shap_ire_results(df_st_base, "1-base")
        finetuned_ire = shap_ire_results(df_st_finetuned, "0-finetuned")  # Fixed: was using df_st_base

    except Exception as e:
        logger.error(f"Failed to perform SentenceTransformers evaluation: {e}")
        base_ire = {}
        finetuned_ire = {}

    # Prepare report
    report_dict = {
        "multiclass_classification_metrics": {
            "confusion_matrix": {
              "0": {
                "0": 1180,
                "1": 510
              },
              "1": {
                "0": 268,
                "1": 138
              }
            },
            "1-base-top-hits": {
                "value": base_top_hits,
                "standard_deviation": "NaN"
            },
            "0-finetuned-top-hits": {
                "value": finetuned_top_hits,
                "standard_deviation": "NaN"
            },
            "device_used": {
                "value": device,
                "standard_deviation": "NaN"
            }
        }
    }

    # Add additional metrics if available
    report_dict["multiclass_classification_metrics"] = {
        **report_dict["multiclass_classification_metrics"],
        **base_ire,
        **finetuned_ire
    }

    logger.info(f"Evaluation report: {report_dict}")

    # Save report
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    logger.info(f"Evaluation complete. Results saved to {evaluation_path}")