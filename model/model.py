from ignite.engine import create_supervised_trainer
from sentence_transformers import SentenceTransformer
from torch import cat, float32, inf, tensor, zeros
from torch.nn import (
    Linear,
    Module,
    MSELoss,
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader


class DataHandler:
    def __init__(self):
        self.string_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 3 * self.string_embedder.get_sentence_embedding_dimension() + 2

    def collate(self, batch):
        """Convert batched data into a tensor to pass to netwwork."""
        x = pad_sequence(
            [
                # Concatenate all fields in data together
                cat(
                    (
                        # Saalary field
                        tensor(
                            [[year["Salary"]] for year in person[:-1]], dtype=float32
                        ),
                        # Year field
                        tensor([[year["Year"]] for year in person[:-1]], dtype=float32),
                        # Bundle all string fields together, encode them, then reshape to unbundle
                        self.string_embedder.encode(
                            [
                                x
                                for year in person[:-1]
                                for x in (
                                    year["Job title"],
                                    year["Employer"],
                                    year["Sector"],
                                )
                            ],
                            convert_to_tensor=True,
                        ).reshape(-1, self.dim - 2),
                    ),
                    dim=1,
                )
                for person in batch
            ],
            batch_first=True,
        )

        y = pad_sequence(
            [tensor([year["Salary"] for year in person[1:]]) for person in batch],
            batch_first=True,
        )

        mask = pad_sequence(
            [zeros(len(person) - 1) for person in batch],
            padding_value=-inf,
            batch_first=True,
        )

        return (x, mask), y

    def dataloader(self, dataset, batch_size=32):
        """Construct a pytorch dataloader for the given dataset."""
        return DataLoader(dataset, batch_size, collate_fn=self.collate)


class Model(Module):
    def __init__(self, in_dim, d_model, n_layers, n_heads, d_ffn=None):
        super().__init__()

        # Default feed-forward network dimension
        if d_ffn is None:
            d_ffn = 4 * d_model

        # Dense layer to convert from input dimension to model dimension
        self.input = Linear(in_dim, d_model)

        # Transformer
        layer = TransformerEncoderLayer(d_model, n_heads, d_ffn, batch_first=True)
        self.transformer = TransformerEncoder(layer, n_layers)

        # Output to get salary estimate from model
        self.output = Linear(d_model, 1)

    def forward(self, x, mask=None):
        size = x.shape[1]
        x = self.input(x)
        x = self.transformer(
            x,
            Transformer.generate_square_subsequent_mask(size),
            src_key_padding_mask=mask,
            is_causal=True,
        )
        x = self.output(x)
        return x.squeeze(-1)


def train(
    dataset,
    batch_size=32,
    learning_rate=0.1,
    epochs=50,
    d_model=64,
    n_layers=4,
    n_heads=4,
    d_ffn=None,
):
    """Train a neural network."""
    data_handler = DataHandler()
    data_loader = data_handler.dataloader(dataset, batch_size)
    model = Model(data_handler.dim, d_model, n_layers, n_heads, d_ffn)
    engine = create_supervised_trainer(
        model,
        optimizer=Adam(model.parameters(), learning_rate),
        loss_fn=MSELoss(),
        model_fn=lambda model, x_and_mask: model(*x_and_mask),
    )
    engine.run(data_loader, epochs)
    return model


def fake_data():
    return [
        [
            {
                "Salary": 1000.0,
                "Year": 2017,
                "Job title": "Software Developer",
                "Employer": "Google",
                "Sector": "Tech",
            },
            {
                "Salary": 1010.0,
                "Year": 2018,
                "Job title": "Software Developer",
                "Employer": "Google",
                "Sector": "Tech",
            },
            {
                "Salary": 1020.0,
                "Year": 2019,
                "Job title": "Software Developer",
                "Employer": "Google",
                "Sector": "Tech",
            },
            {
                "Salary": 1030.0,
                "Year": 2020,
                "Job title": "Software Developer",
                "Employer": "Google",
                "Sector": "Tech",
            },
        ],
        [
            {
                "Salary": 1000.0,
                "Year": 2016,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
            {
                "Salary": 1020.0,
                "Year": 2017,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
            {
                "Salary": 1040.0,
                "Year": 2018,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
            {
                "Salary": 1060.0,
                "Year": 2019,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
            {
                "Salary": 1080.0,
                "Year": 2020,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
            {
                "Salary": 1100.0,
                "Year": 2021,
                "Job title": "Software Developer",
                "Employer": "Facebook",
                "Sector": "Tech",
            },
        ],
    ]
