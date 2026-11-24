import torch
from .model import Config, TinyMamba2


def _m():
    torch.manual_seed(0)
    return TinyMamba2(Config()).eval()


def test_cache_shapes():
    m, c = _m(), Config()
    ids = torch.randint(1, 63, (3, 10))
    lg, ca = m(ids, return_cache=True)
    B, T, H, P, N = 3, 10, c.n_head, c.head_dim, c.d_state
    assert lg.shape == (B, T, c.vocab)
    exp = {"h": (B, T, H, P, N), "a": (B, T, H), "dt": (B, T, H), "B": (B, T, N), "C": (B, T, N), "x": (B, T, H, P), "y": (B, T, H, P)}
    for k, s in exp.items():
        assert len(ca[k]) == c.n_layer and all(tuple(v.shape) == s for v in ca[k]), k
    assert len(ca["resid"]) == c.n_layer + 1 and all(v.shape == (B, T, c.d_model) for v in ca["resid"])
    assert ((ca["a"][0] > 0) & (ca["a"][0] < 1)).all()


def test_identity_hook():
    m = _m()
    ids = torch.randint(1, 63, (2, 12))
    a, _ = m(ids)
    b, _ = m(ids, state_hook=lambda l, t, h: h)
    assert torch.equal(a, b)


def test_zero_hook_changes_later():
    m = _m()
    ids = torch.randint(1, 63, (2, 12))
    a, _ = m(ids)
    b, _ = m(ids, state_hook=lambda l, t, h: torch.zeros_like(h) if (l == 0 and t == 3) else h)
    assert torch.allclose(a[:, :3], b[:, :3]) and not torch.allclose(a[:, 5:], b[:, 5:])


def test_grad_to_delta():
    m = _m()
    ids = torch.randint(1, 63, (2, 12))
    delta = torch.zeros(2, 4, 16, 8, requires_grad=True)
    lg, _ = m(ids, state_hook=lambda l, t, h: h + delta if (l == 0 and t == 3) else h)
    lg[:, -1].sum().backward()
    assert delta.grad.abs().sum() > 0
