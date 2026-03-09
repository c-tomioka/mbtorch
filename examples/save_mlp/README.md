# save_mlp

Train a 2-layer MLP with SGD, then serialize and deserialize it using both JSON and Binary `.mbt` formats. Verify that the restored models produce the same output as the original.

## Run

```bash
moon run examples/save_mlp
```

## Expected output

```
Training: loss 137.27... -> 5.06...
JSON .mbt size: ... chars
Binary .mbt size: ... bytes
Original vs JSON:   max diff = 0
Original vs Binary: max diff = ...e-7
Roundtrip OK!
```

- **JSON `.mbt`**: Lossless roundtrip (diff = 0) since values are stored as Double text.
- **Binary `.mbt`**: Near-lossless (diff ≈ 1e-7) due to Double → float32 → Double conversion.
- Both formats are interconvertible and produce functionally equivalent models.
