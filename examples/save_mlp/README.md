# save_mlp

Train a 2-layer MLP with SGD, then serialize and deserialize it using MbTorch's `.mbt`-style JSON format to verify that the restored model produces the same output.

## Run

```bash
moon run examples/save_mlp
```

## Expected output

```
Training: loss 137.27... -> 5.06...
Serialized JSON length: ... chars
Max output difference after roundtrip: 0
Roundtrip OK!
```

Loss decreases during training, and the deserialized model produces identical outputs to the original.
