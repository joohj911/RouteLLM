import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from routellm.routers.matrix_factorization.model import MODEL_IDS

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.models_a = torch.tensor(
            [MODEL_IDS[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [MODEL_IDS[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompt_id = [sample["idx"] for sample in data]
        self.winners = [sample["winner"] for sample in data]

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        assert self.winners[index] in ["model_a", "model_b"], self.winners[index]
        if self.winners[index] == "model_a":
            return self.models_a[index], self.models_b[index], self.prompt_id[index]
        else:
            return self.models_b[index], self.models_a[index], self.prompt_id[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)


class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models,
        text_dim=384,
        num_classes=1,
        use_proj=True,
        npy_path=None,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        # num_prompts는 embeddings.npy 전체 크기에서 자동으로 결정.
        # train_data.json은 필터링된 subset이지만 idx는 원본 embeddings를 가리킴.
        embeddings = np.load(npy_path)
        num_prompts = embeddings.shape[0]
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float32))

        if self.use_proj:
            # Sequential wrapper matches MFModel's state dict keys (text_proj.0.weight)
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        # Sequential wrapper matches MFModel's state dict keys (classifier.0.weight)
        self.classifier = torch.nn.Sequential(nn.Linear(dim, num_classes, bias=False))

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_win, model_loss, prompt, test=False, alpha=0.05):
        model_win = model_win.to(self.get_device())
        model_loss = model_loss.to(self.get_device())
        prompt = prompt.to(self.get_device())

        model_win_embed = self.P(model_win)
        model_win_embed = F.normalize(model_win_embed, p=2, dim=1)
        model_loss_embed = self.P(model_loss)
        model_loss_embed = F.normalize(model_loss_embed, p=2, dim=1)
        prompt_embed = self.Q(prompt)
        if not test:
            prompt_embed = prompt_embed + torch.randn_like(prompt_embed) * alpha
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(
            (model_win_embed - model_loss_embed) * prompt_embed
        ).squeeze()

    @torch.no_grad()
    def predict(self, model_win, model_loss, prompt):
        logits = self.forward(model_win, model_loss, prompt, test=True)
        return logits > 0


def evaluator(net, test_iter, device):
    net.eval()
    ls_fn = nn.BCEWithLogitsLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models_a, models_b, prompts in test_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            logits = net(models_a, models_b, prompts)
            labels = torch.ones_like(logits)
            loss = ls_fn(logits, labels)
            pred_labels = net.predict(models_a, models_b, prompts)

            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())
            num_samples += labels.shape[0]

    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples


def train_loops(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=evaluator,
    **kwargs,
):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss(reduction="mean")

    best_test_acc = -1

    def train_epoch():
        net.train()
        train_loss_sum, n = 0.0, 0
        for models_a, models_b, prompts in train_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            output = net(models_a, models_b, prompts, alpha=alpha)
            ls = loss(output, torch.ones_like(output))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            train_loss_sum += ls.item() * len(models_a)
            n += len(models_a)
        return train_loss_sum / n

    train_losses = []
    test_losses = []
    test_acces = []
    progress_bar = tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        train_ls = train_epoch()
        train_losses.append(train_ls)
        info = {"train_loss": train_ls, "epoch": epoch}

        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, device)
            test_losses.append(test_ls)
            test_acces.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            info.update(
                {
                    "test_loss": test_ls,
                    "test_acc": test_acc,
                    "best_test_acc": best_test_acc,
                }
            )

        progress_bar.set_postfix(**info)
        progress_bar.update(1)

    progress_bar.close()
    return best_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MF router on BFCL data.")
    parser.add_argument(
        "--train-data", type=str, required=True,
        help="Path to train_data.json from prepare_bfcl_data.py convert"
    )
    parser.add_argument(
        "--npy-path", type=str, required=True,
        help="Path to embeddings.npy from prepare_bfcl_data.py embed"
    )
    parser.add_argument(
        "--output-path", type=str, default="./bfcl_mf_model.pt",
        help="Where to save the trained model checkpoint"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--text-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--no-proj", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of train_data to use as internal validation set")
    args = parser.parse_args()

    data = json.load(open(args.train_data))
    filtered_data = [
        s for s in data
        if s["winner"] in ["model_a", "model_b"] and s["model_a"] != s["model_b"]
    ]
    print(f"Loaded {len(filtered_data)} samples (filtered from {len(data)})")

    random.shuffle(filtered_data)
    n_val = max(1, int(len(filtered_data) * args.val_ratio))
    train_data = filtered_data[:-n_val]
    val_data = filtered_data[-n_val:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    train_loader = PairwiseDataset(train_data).get_dataloaders(args.batch_size, shuffle=True)
    val_loader = PairwiseDataset(val_data).get_dataloaders(1024, shuffle=False)

    model = MFModel_Train(
        dim=args.dim,
        num_models=len(MODEL_IDS),
        text_dim=args.text_dim,
        use_proj=not args.no_proj,
        npy_path=args.npy_path,
    ).to(args.device)

    best_acc = train_loops(
        model,
        train_loader,
        val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        num_epochs=args.num_epochs,
        device=args.device,
    )

    # Q (prompt embeddings)는 추론 시 불필요하므로 제외하여 저장
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("Q.")}
    torch.save(state, args.output_path)
    print(f"\nSaved model → {args.output_path}  (best_val_acc={best_acc:.4f})")
