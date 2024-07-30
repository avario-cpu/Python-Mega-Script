import json
import torch
from sentence_transformers import SentenceTransformer, util
import time
from src.robeau.core.constants import ROBEAU_PROMPTS_JSON_FILE_PATH as ROBEAU_PROMPTS
from typing import Optional


class SBERTMatcher:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        file_path=None,
        similarity_threshold=0.6,
    ):
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.embeddings = self.load_embeddings(file_path) if file_path else {}
        self.metadata = self.load_metadata(file_path) if file_path else {}
        self.similarity_threshold = similarity_threshold

    def load_embeddings(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        embeddings: dict = {}
        for section, items in data.items():
            embeddings[section] = []
            for item in items:
                main_text = item["text"]
                all_texts = [main_text] + item.get("synonyms", [])
                for text in all_texts:
                    embedding = self.model.encode(text, convert_to_tensor=True)
                    if torch.cuda.is_available() and isinstance(
                        embedding, torch.Tensor
                    ):
                        embedding = embedding.to("cuda")
                    embeddings[section].append((main_text, text, embedding))
        return embeddings

    def load_metadata(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        metadata: dict = {}
        for section, items in data.items():
            metadata[section] = []
            for item in items:
                main_text = item["text"]
                rudeness_points = item.get("rudeness_points", 0)
                all_texts = [(main_text, rudeness_points)] + [
                    (syn, rudeness_points) for syn in item.get("synonyms", [])
                ]
                for text, rudeness_pts in all_texts:
                    metadata[section].append((main_text, text, rudeness_pts))
        return metadata

    def measure_similarity(self, embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def check_for_best_matching_synonym(
        self,
        message: str,
        show_details: bool = False,
        labels: Optional[list] = None,
        stop_command: Optional[bool] = False,
    ):
        start_time = time.time()
        input_embedding = self.model.encode(message, convert_to_tensor=True)
        if torch.cuda.is_available() and isinstance(input_embedding, torch.Tensor):
            input_embedding = input_embedding.to("cuda")

        max_similarity = -1
        best_match = None
        best_synonym = None

        categories = labels if labels else self.embeddings.keys()

        for category in categories:
            items = self.embeddings.get(category, [])
            metadata_items = self.metadata.get(category, [])
            for (main_text, text, embedding), (_, _, rudeness_points) in zip(
                items, metadata_items
            ):
                similarity = self.measure_similarity(input_embedding, embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = main_text
                    best_synonym = text

        end_time = time.time()
        inference_time = end_time - start_time

        if show_details:
            print(
                f"Input: <{message}> has match value <{max_similarity:.3f}> from matching with <{best_synonym}> for original text: <{best_match}> (exec.time: {inference_time:.4f})"
            )

        if max_similarity < self.similarity_threshold:
            return (None, 0) if stop_command else None
        else:
            return (best_match, rudeness_points) if stop_command else best_match


def main():
    matcher = SBERTMatcher(file_path=ROBEAU_PROMPTS)
    while True:
        message = input("Enter a message (or type 'exit' to quit): ")
        if message.lower() == "exit":
            break

        best_match = matcher.check_for_best_matching_synonym(
            message, show_details=True, labels=["Greeting"]
        )
        print(f"Best Match: {best_match}\n")


if __name__ == "__main__":
    main()