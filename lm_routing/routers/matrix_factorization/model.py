import torch
from huggingface_hub import PyTorchModelHubMixin
from sentence_transformers import SentenceTransformer



EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def build_classifier(dim: int, num_classes: int, mlp_hidden: int = 0):
    """
    분류기 head 생성. mlp_hidden=0 이면 기존 선형(bias 없음), >0 이면 1-hidden MLP.

    학습(MFModel_Train)과 추론(MFModel)이 동일한 빌더를 써야 state_dict 키가 일치한다.
    """
    if mlp_hidden and mlp_hidden > 0:
        return torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, num_classes, bias=False),
        )
    return torch.nn.Sequential(torch.nn.Linear(dim, num_classes, bias=False))


class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim=384,
        num_classes=1,
        use_proj=True,
        mlp_hidden=0,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = build_classifier(dim, num_classes, mlp_hidden)

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        # multilingual-e5-small uses "query: " prefix for asymmetric retrieval tasks
        prompt_embed = get_embedding_model().encode(
            f"query: {prompt}",
            convert_to_tensor=True,
            device=str(self.get_device()),
        )
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
