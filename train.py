"""
Interleaved carry-propagation transformer.

1. Interleaved format: [s_0, c_0, s_1, c_1, ..., s_10, c_10]
   - s_i = a_i + b_i (digit sum, 0-18)
   - c_i = output digit (0-9)
   Now c_i only needs to attend 0-2 positions back.

2. Same matched position token indexing, but instead takes the sum to reduce token count (0 to 18)

2. Loss only on output positions (c_i), skip digit-sum positions.

3. Sinusoidal pos encoding to save params (no learned pos needed).

4. Quick sweep: 500k examples, 2000 epochs per config.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np, math, time, json, os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

device = 'cuda'
MAX_DIGITS = 10
VOCAB = 19  # 0-18 for digit sums; 0-9 also used for output digits
OUT_DIGITS = MAX_DIGITS + 1  # 11
SEQ_LEN = 2 * OUT_DIGITS  # 22 (interleaved: s_0,c_0,s_1,c_1,...,s_10,c_10)

def encode_rev(n, l):
    d = []
    for _ in range(l): d.append(n%10); n//=10
    return d

def preprocess(a, b):
    """Compute per-digit sums (no carry). Returns 11 values in [0..18]."""
    ad = encode_rev(a, MAX_DIGITS)
    bd = encode_rev(b, MAX_DIGITS)
    return [ad[i]+bd[i] for i in range(MAX_DIGITS)] + [0]

def make_example(a, b):
    """Interleaved: s_0, c_0, s_1, c_1, ..., s_10, c_10"""
    sums = preprocess(a, b)
    c_digits = encode_rev(a+b, OUT_DIGITS)
    seq = []
    for i in range(OUT_DIGITS):
        seq.append(sums[i])
        seq.append(c_digits[i])
    return seq

def postprocess(output_tokens):
    """Convert output digits (reversed) to integer."""
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


# Loss positions: predict c_i at even target positions
# input  = seq[0:21], target = seq[1:22]
# target[0] = c_0 (predict), target[1] = s_1 (skip),
# target[2] = c_1 (predict), target[3] = s_2 (skip), ...
# target[20] = c_10 (predict)
# So loss_positions = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
LOSS_POS = list(range(0, 21, 2))  # 11 positions


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
            # Select only output positions
            lo_sel = lo[:, loss_idx].reshape(-1, VOCAB)
            tgt_sel = tgt[:, loss_idx].reshape(-1)
            loss = F.cross_entropy(lo_sel, tgt_sel)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            el += loss.item(); nb += 1

        freq = 100 if ep > 1000 else 50
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
            if ex >= 99.0:
                print(f"  *** TARGET: {ex:.2f}% ***"); break

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


if __name__ == '__main__':
    save_dir = '/content/addition_experiment_claude_ver2'
    os.makedirs(save_dir, exist_ok=True)

    print("Creating datasets (interleaved carry-prop)...")
    train_ds = DS(500000, 42)  # 500k for faster sweeps
    test_data = DS(10000, 9999).data.to(device)

    # Verify
    ex = train_ds.data[0].tolist()
    a, b = train_ds.ab[0]
    print(f"a={a}, b={b}, a+b={a+b}")
    print(f"Interleaved: {ex}")
    out_digits = [ex[2*i+1] for i in range(OUT_DIGITS)]
    print(f"Output digits: {out_digits}")
    print(f"Decoded: {postprocess(out_digits)}")
    assert postprocess(out_digits) == a + b, "Verification failed!"
    print("OK!\n")

    results = {}

    # Param formula (sinusoidal, no learned pos):
    # P = 19d + nL*(4d^2 + 4d + 2d*ff) + 19
    configs = [
        # (name, d, ff, nL, nH, lr, epochs, bs)

        # Validation
        ('val_d16', 16, 32, 2, 4, 1e-3, 100, 4096),

        # === Under 512 params ===
        # d=8, ff=3, 1L: 459+48=507
        ('d8f3_1L_1h', 8, 3, 1, 1, 5e-3, 3000, 16384),
        ('d8f3_1L_4h', 8, 3, 1, 4, 5e-3, 3000, 16384),

        # d=6, ff=17, 1L: 301+204=505
        ('d6f17_1L_1h', 6, 17, 1, 1, 5e-3, 3000, 16384),
        ('d6f17_1L_3h', 6, 17, 1, 3, 5e-3, 3000, 16384),

        # d=5, ff=27, 1L: 234+270=504
        ('d5f27_1L_1h', 5, 27, 1, 1, 5e-3, 3000, 16384),

        # d=4, ff=42, 1L: 175+336=511
        ('d4f42_1L_1h', 4, 42, 1, 1, 5e-3, 3000, 16384),
        ('d4f42_1L_2h', 4, 42, 1, 2, 5e-3, 3000, 16384),

        # d=4, ff=16, 2L: 255+256=511
        ('d4f16_2L_1h', 4, 16, 2, 1, 5e-3, 3000, 16384),
    ]

    for cname, d, ff, nl, nh, lr, ep, bs in configs:
        m = TinyCarryTransformer(d, ff, nl, nh).to(device)
        pc = m.count_params()
        tag = f"<=512" if pc <= 512 else f"OVER({pc})"
        print(f"\n>>> {cname}: {pc} params [{tag}]")
        a = train_run(m, cname, lr, ep, bs, train_ds, test_data, save_dir)
        results[cname] = {'p': pc, 'a': a}

        if a >= 99.0 and pc <= 512:
            print(f"\n{'='*55}")
            print(f"WINNER: {cname} — {pc} params, {a:.2f}%")
            print(f"{'='*55}")
            # Final eval on fresh 10k
            m.load_state_dict(torch.load(os.path.join(save_dir, f'{cname}_best.pt'),
                                         weights_only=True))
            m.eval()
            final_ds = DS(10000, 77777)
            fd = final_ds.data.to(device)
            exact_ok = 0; total = 0; failures = []
            lp = torch.tensor(LOSS_POS, device=device)
            with torch.no_grad():
                for i in range(0, len(fd), 2048):
                    c = fd[i:i+2048]
                    lo = m(c[:, :-1])
                    p = lo[:, lp].argmax(-1)
                    t = c[:, 1:][:, lp]
                    for j in range(len(c)):
                        pj, tj = p[j].cpu().tolist(), t[j].cpu().tolist()
                        if pj == tj:
                            exact_ok += 1
                        elif len(failures) < 30:
                            av, bv = final_ds.ab[i+j]
                            failures.append({
                                'a': int(av), 'b': int(bv),
                                'exp': int(av)+int(bv),
                                'pred': postprocess(pj)
                            })
                        total += 1
            acc = exact_ok / total * 100
            print(f"Final eval: {acc:.2f}% ({exact_ok}/{total})")
            if failures:
                print("Sample failures:")
                for f in failures[:10]:
                    print(f"  {f['a']}+{f['b']}={f['exp']} pred:{f['pred']}")
            with open(os.path.join(save_dir, f'{cname}_final.json'), 'w') as f:
                json.dump({'p': pc, 'acc': acc, 'failures': failures[:20]}, f, indent=2)
            break

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    for k, v in sorted(results.items(), key=lambda x: (-x[1]['a'], x[1]['p'])):
        m2 = "Y" if v['p'] <= 512 else "N"
        print(f"  [{m2}] {k}: {v['p']}p, {v['a']:.2f}%")
