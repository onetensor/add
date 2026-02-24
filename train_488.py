"""
same as before, just testing these:
d=7, ff=8: 488p (below 491!)
d=7, ff=5: 446p
d=7, ff=3: 418p
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np, math, time, json, os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

device = 'cuda'
MAX_DIGITS = 10; VOCAB = 19; OUT_DIGITS = 11; SEQ_LEN = 22

def encode_rev(n, l):
    d = []
    for _ in range(l): d.append(n%10); n//=10
    return d

def preprocess(a, b):
    ad = encode_rev(a, MAX_DIGITS)
    bd = encode_rev(b, MAX_DIGITS)
    return [ad[i]+bd[i] for i in range(MAX_DIGITS)] + [0]

def make_example(a, b):
    sums = preprocess(a, b)
    c_digits = encode_rev(a+b, OUT_DIGITS)
    seq = []
    for i in range(OUT_DIGITS):
        seq.append(sums[i])
        seq.append(c_digits[i])
    return seq

def postprocess(output_tokens):
    d = [t for t in output_tokens if 0 <= t <= 9]
    while len(d) > 1 and d[-1] == 0: d.pop()
    return int(''.join(str(x) for x in reversed(d)))

class DS(Dataset):
    def __init__(self, n, seed=42):
        rng = np.random.RandomState(seed); mx = 10**MAX_DIGITS - 1
        a = rng.randint(0, mx+1, n); b = rng.randint(0, mx+1, n)
        self.data = torch.tensor([make_example(int(x),int(y)) for x,y in zip(a,b)], dtype=torch.long)
        self.ab = list(zip(a.tolist(), b.tolist()))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def sinusoidal_pe(seq_len, d, base=22.0):
    pe = torch.zeros(seq_len, d)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(base) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d//2])
    return pe

class TinyCarryTransformer(nn.Module):
    def __init__(self, d, d_ff, n_layers, n_heads):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.register_buffer('pe', sinusoidal_pe(SEQ_LEN, d, base=22.0))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False),
                'n1': nn.LayerNorm(d),
                'f1': nn.Linear(d, d_ff, bias=False),
                'f2': nn.Linear(d_ff, d, bias=False),
                'n2': nn.LayerNorm(d),
            }))
        self.ob = nn.Parameter(torch.zeros(VOCAB))

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pe[:T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for l in self.layers:
            a, _ = l['attn'](h, h, h, attn_mask=mask)
            h = l['n1'](h + a)
            h = l['n2'](h + l['f2'](F.gelu(l['f1'](h))))
        return F.linear(h, self.tok.weight, self.ob)

    def count_params(self): return sum(p.numel() for p in self.parameters())

LOSS_POS = list(range(0, 21, 2))

def train_run(model, name, lr, epochs, bs, train_ds, test_data, save_dir):
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2)

    pc = model.count_params()
    print(f"\n{'='*55}")
    print(f"{name}: {pc} params | lr={lr} ep={epochs} bs={bs}")
    print(f"{'='*55}")

    best = 0; t0 = time.time()
    hist = {'ep':[],'tl':[],'ta':[],'ea':[]}
    loss_idx = torch.tensor(LOSS_POS, device=device)

    for ep in range(1, epochs+1):
        model.train(); el = 0; nb = 0
        for batch in loader:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            lo = model(inp)
            lo_sel = lo[:, loss_idx].reshape(-1, VOCAB)
            tgt_sel = tgt[:, loss_idx].reshape(-1)
            loss = F.cross_entropy(lo_sel, tgt_sel)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            el += loss.item(); nb += 1

        freq = 100 if ep > 2000 else 50
        do = ep % freq == 0 or ep <= 5 or ep == epochs
        if do:
            model.eval()
            with torch.no_grad():
                pa = []; ta2 = []
                for i in range(0, len(test_data), 4096):
                    c = test_data[i:i+4096]
                    lo = model(c[:, :-1])
                    pa.append(lo[:, loss_idx].argmax(-1))
                    ta2.append(c[:, 1:][:, loss_idx])
                preds = torch.cat(pa); targets = torch.cat(ta2)
                tok = (preds == targets).float().mean().item() * 100
                ex = (preds == targets).all(-1).float().mean().item() * 100
            hist['ep'].append(ep); hist['tl'].append(el/nb)
            hist['ta'].append(tok); hist['ea'].append(ex)
            m = ""
            if ex > best:
                best = ex; m = " ***"
                torch.save(model.state_dict(), os.path.join(save_dir, f'{name}_best.pt'))
            print(f"  [{time.time()-t0:5.0f}s] Ep {ep:5d} | L {el/nb:.4f} | "
                  f"Tok {tok:6.2f}% | Ex {ex:6.2f}%{m}")
            sys.stdout.flush()
            if ex >= 99.5:
                print(f"  *** EXCELLENT: {ex:.2f}% ***"); break

    with open(os.path.join(save_dir, f'{name}_hist.json'), 'w') as f:
        json.dump(hist, f)
    if len(hist['ep']) > 1:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(hist['ep'], hist['tl']); ax[0].set_yscale('log'); ax[0].set_title('Loss')
        ax[1].plot(hist['ep'], hist['ea']); ax[1].axhline(99, c='r', ls='--'); ax[1].set_title('ExAcc%')
        ax[2].plot(hist['ep'], hist['ta']); ax[2].set_title('TokAcc%')
        plt.suptitle(f'{name}: {pc}p, best {best:.1f}%')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_curves.png'), dpi=150)
        plt.close()
    print(f"  Best: {best:.2f}%")
    return best


def final_eval(model, name, save_dir):
    """Run full autoregressive eval."""
    model.eval()
    rng = np.random.RandomState(77777)
    mx = 10**MAX_DIGITS - 1
    n_test = 10000
    a_arr = rng.randint(0, mx+1, n_test)
    b_arr = rng.randint(0, mx+1, n_test)

    test_data = torch.tensor([make_example(int(x), int(y))
                              for x, y in zip(a_arr, b_arr)], dtype=torch.long).to(device)

    lp = torch.tensor(LOSS_POS, device=device)
    exact_ok = 0; total = 0; failures = []

    with torch.no_grad():
        for i in range(0, len(test_data), 2048):
            c = test_data[i:i+2048]
            lo = model(c[:, :-1])
            p = lo[:, lp].argmax(-1)
            t = c[:, 1:][:, lp]
            for j in range(len(c)):
                pj, tj = p[j].cpu().tolist(), t[j].cpu().tolist()
                if pj == tj:
                    exact_ok += 1
                elif len(failures) < 30:
                    av, bv = int(a_arr[i+j]), int(b_arr[i+j])
                    failures.append({'a': av, 'b': bv, 'exp': av+bv, 'pred': postprocess(pj)})
                total += 1
    acc = exact_ok / total * 100
    print(f"  TF eval: {acc:.2f}% ({exact_ok}/{total})")

    # Autoregressive
    n_ar = 1000
    ar_ok = 0
    ar_rng = np.random.RandomState(12345)
    ar_a = ar_rng.randint(0, mx+1, n_ar)
    ar_b = ar_rng.randint(0, mx+1, n_ar)
    for idx in range(n_ar):
        av, bv = int(ar_a[idx]), int(ar_b[idx])
        sums = preprocess(av, bv)
        seq = [sums[0]]
        for i in range(OUT_DIGITS):
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                lo = model(inp)
            c_i = lo[0, -1, :10].argmax().item()
            seq.append(c_i)
            if i < OUT_DIGITS - 1:
                seq.append(sums[i+1])
        pred_digits = [seq[2*i+1] for i in range(OUT_DIGITS)]
        if postprocess(pred_digits) == av + bv:
            ar_ok += 1
    ar_acc = ar_ok / n_ar * 100
    print(f"  AR eval: {ar_acc:.2f}% ({ar_ok}/{n_ar})")
    return acc, ar_acc


if __name__ == '__main__':
    save_dir = '/content/addition_experiment_claude_ver2'
    os.makedirs(save_dir, exist_ok=True)

    print("Creating datasets...")
    train_ds = DS(2000000, 42)  # 2M for best results
    test_data = DS(10000, 9999).data.to(device)

    results = {}

    configs = [
        # (name, d, ff, nL, nH, lr, epochs, bs)
        # d=7 configs — all below 491
        ('d7f8_1L_1h', 7, 8, 1, 1, 3e-3, 3000, 16384),  # 488p
        ('d7f5_1L_1h', 7, 5, 1, 1, 3e-3, 5000, 16384),  # 446p
        ('d7f3_1L_1h', 7, 3, 1, 1, 3e-3, 5000, 16384),  # 418p

        # Also try d=7 with higher lr
        ('d7f8_1L_1h_hlr', 7, 8, 1, 1, 5e-3, 3000, 16384),  # 488p

        # d=8, ff=2: 459+32=491 — exactly at boundary
        ('d8f2_1L_1h', 8, 2, 1, 1, 3e-3, 3000, 16384),  # 491p
    ]

    for cname, d, ff, nl, nh, lr, ep, bs in configs:
        m = TinyCarryTransformer(d, ff, nl, nh).to(device)
        pc = m.count_params()
        print(f"\n>>> {cname}: {pc} params")
        a = train_run(m, cname, lr, ep, bs, train_ds, test_data, save_dir)
        results[cname] = {'p': pc, 'a': a}

        if a >= 99.0:
            m.load_state_dict(torch.load(os.path.join(save_dir, f'{cname}_best.pt'),
                                         weights_only=True))
            tf_acc, ar_acc = final_eval(m, cname, save_dir)
            results[cname]['tf_final'] = tf_acc
            results[cname]['ar_final'] = ar_acc

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    for k, v in sorted(results.items(), key=lambda x: (-x[1]['a'], x[1]['p'])):
        extra = ""
        if 'tf_final' in v:
            extra = f" | final: {v['tf_final']:.2f}% TF, {v['ar_final']:.2f}% AR"
        print(f"  {k}: {v['p']}p, train_best: {v['a']:.2f}%{extra}")
