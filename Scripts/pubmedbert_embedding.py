# scripts/pubmedbert_embedding.py

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class PubMedBERTEmbedder:
    def __init__(self, model_name="NeuML/pubmedbert-base-embeddings", device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            last_hidden = outputs.last_hidden_state  # (1, 512, 768)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
            masked = last_hidden * attention_mask
            summed = masked.sum(dim=1)
            count = attention_mask.sum(dim=1)
            mean_pooled = summed / count  # (1, 768)
            return mean_pooled.squeeze(0).cpu().numpy()

    def embed_batch(self, texts):
        return np.vstack([self.embed(text) for text in texts])
