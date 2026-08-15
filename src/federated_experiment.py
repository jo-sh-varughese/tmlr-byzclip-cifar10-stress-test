"""Shared federated-training harness wiring data/models/attacks into ByzClip21SGD2M.

Design note on how attacks map onto Algorithm 1's client roles:
  - IPM is a vector-injection attack: Byzantine clients bypass the honest v_i/g_i/clip
    pipeline entirely and transmit an arbitrary crafted c_i (per Algorithm 1's "for
    Byzantine i: c_i = arbitrary_vector()" line). This is implemented via the
    `byzantine_fn` hook in ByzClip21SGD2M.step, operating on the round's honest
    transmitted c_i vectors (omniscient-coalition attack model).
  - Label-flipping is a DATA-poisoning attack, not a protocol deviation: the attacker
    follows the honest v_i/g_i/clip/noise pipeline exactly, but computes its local
    stochastic gradient on systematically mislabeled data. Mechanically this means
    label-flip attackers are simulated as additional "honest-pipeline" clients (they go
    through ByzClip21SGD2M's honest v_i update) whose data loader applies
    attacks.apply_label_flip to every label before computing the loss. There is no
    n_byzantine > 0 in the ByzClip21SGD2M sense for this attack type.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from byz_clip21_sgd2m import ByzClip21SGD2M
from byz_clip_sgd import ByzClipSGD
from safe_dshb import SafeDSHB
from robust_aggregators import apply_ragg
from attacks import ipm_byzantine_batch, apply_label_flip
from data import make_client_loaders, InfiniteLoaderIter, partition_iid


def compute_sigma_omega(tau, epsilon, T, delta=1e-5):
    """sigma_omega = (tau/epsilon) * sqrt(T * log(1/delta)), per the task spec.

    Privacy amplification by sub-sampling is deliberately NOT applied, matching the
    paper's stated design choice ("we disable privacy amplification by sub-sampling").
    """
    if epsilon is None or epsilon == float("inf"):
        return 0.0
    return (tau / epsilon) * math.sqrt(T * math.log(1.0 / delta))


class FlatModel:
    """Wraps an nn.Module to expose flat-vector get/set of parameters and flat gradients."""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.d = sum(p.numel() for p in model.parameters())

    def get_flat(self):
        return nn_utils.parameters_to_vector(self.model.parameters()).detach().clone()

    def set_flat(self, x):
        nn_utils.vector_to_parameters(x, self.model.parameters())

    def flat_grad(self, x, xb, yb, loss_fn):
        self.set_flat(x)
        self.model.zero_grad()
        out = self.model(xb.to(self.device))
        loss = loss_fn(out, yb.to(self.device))
        loss.backward()
        grad = nn_utils.parameters_to_vector([p.grad for p in self.model.parameters()]).detach().clone()
        return grad, loss.item()

    @torch.no_grad()
    def evaluate(self, x, loader):
        self.set_flat(x)
        self.model.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            out = self.model(xb.to(self.device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
        self.model.train()
        return correct / total


def run_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    ablation=None,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Run one full Byz-Clip21-SGD2M training run and return a results dict.

    `n_regular` regular (never-poisoned) clients always exist. For attack_type="ipm",
    `n_byzantine` additional vector-injection attackers are added (algorithm's B set).
    For attack_type="label_flip", `n_byzantine` additional clients are added to the
    HONEST pipeline but trained on flipped labels (see module docstring). For
    attack_type=None, n_byzantine is ignored (treated as 0).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0
    if ablation == "no_clip_no_dp":
        sigma_omega = 0.0  # tau forced to inf inside ByzClip21SGD2M too

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = ByzClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device, ablation=ablation,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed,
            "ablation": ablation, "d": d,
        },
    }


def run_baseline_experiment(
    algo_name,            # "byz_clip_sgd" | "safe_dshb"
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    beta=0.1,             # Safe-DSHB's client-momentum, per the source paper Sec. 6:
                           # "For Byz-Clip21-SGD2M and Safe-DSHB, we fix ... beta=0.1".
                           # Ignored for Byz-Clip-SGD (it has no momentum parameter).
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Run one full external-baseline (Byz-Clip-SGD or Safe-DSHB, arXiv:2603.23472
    Appendix Algorithm 3 / Algorithm 4) training run, under the identical harness
    (data partitioning, attack simulation, DP-noise calibration, evaluation) already
    used for Byz-Clip21-SGD2M in `run_experiment`, so results are directly comparable.
    """
    if algo_name not in ("byz_clip_sgd", "safe_dshb"):
        raise ValueError(f"Unknown algo_name '{algo_name}'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    if algo_name == "byz_clip_sgd":
        algo = ByzClipSGD(
            d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
            gamma=gamma, tau=tau, sigma_omega=sigma_omega, ragg_fn=ragg_fn, device=device,
        )
    else:
        algo = SafeDSHB(
            d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
            beta=beta, gamma=gamma, tau=tau, sigma_omega=sigma_omega, ragg_fn=ragg_fn, device=device,
        )

    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "algo_name": algo_name,
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta if algo_name == "safe_dshb" else None,
            "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }
