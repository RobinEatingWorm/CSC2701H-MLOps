from torch.nn import Module, Linear, TransformerEncoder, TransformerEncoderLayer
from sentence_transformers import SentenceTransformer


class Model(Module):
    def __init__(self, d_model, n_layers, n_heads, d_ffn=None):
        super().__init__()

        # Default feed-forward network dimension
        if d_ffn is None:
            d_ffn = 4 * d_model

        # Preprocessor to turn string fields to vectors
        self.preprocessor = SentenceTransformer("all-MiniLM-L6-v2")

        # Dense layer to convert from input dimension to model dimension
        in_dim = self.preprocessor.get_sentence_embedding_dimension() * 3 + 2
        self.input = Linear(in_dim, d_model)

        # Transformer
        layer = TransformerEncoderLayer(d_model, n_heads, d_ffn, batch_first=True)
        self.transformer = TransformerEncoder(layer, n_layers)

        # Output to get salary estimate from model
        self.output = Linear(d_model, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.transformer(x, is_causal=True)
        x = self.output(x)
        return x
