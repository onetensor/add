## How it works

Preprocessing: given `A + B`, compute per-column digit sums `s_i = a_i + b_i` (values 0–18). No carries, just raw sums. This is a pure function with no learned components.

Interleaving: the sequence alternates digit sums and output slots:

```
[s_0, c_0, s_1, c_1, ..., s_10, c_10]
```

where `s_i` is the precomputed digit sum (given as input) and `c_i` is the output digit (predicted by the model). Digits are reversed (LSB first) so carries flow left to right during autoregressive generation.

To predict `c_i`, the model only needs to look at `s_i` (current column sum, 1 position back) and `c_{i-1}` (previous output digit, 2 positions back — which implicitly encodes the incoming carry). The long-range carry propagation problem becomes a purely local, 2-token lookback at every step.

So, the transformer only has to learn a lookup table of around 38 entries. Given `s_i` (0–18) and an incoming carry (0 or 1, inferable from `c_{i-1}` and `s_{i-1}`), output `(s_i + carry) mod 10`. That's it.

It uses 507 params and takes 250 epochs for this model to get over 99% val accuracy on the hold out validation set of 10k examples.
