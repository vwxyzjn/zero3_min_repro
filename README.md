# get started

This repo reproduces a bug with the latest transformers, accelerate, deepspeed and datasets. The bug is that the logits obtained from `.generate(return_logits=True, ...)` is not the same as the logits obtained from `.forward(query_responses)`, **when using deepspeed**.


```
pip install -r requirements.txt
```

# run the following to get the logs


```bash
python min_repro.py > logs/1gpu.log 2>&1
accelerate launch --num_processes 8  min_repro.py > logs/8gpu.log 2>&1
accelerate launch --config_file deepspeed3.yaml  min_repro.py --deepspeed3 > logs/deepspeed3.log 2>&1
accelerate launch --config_file deepspeed2.yaml  min_repro.py --deepspeed2 > logs/deepspeed2.log 2>&1
```

# Results

Feel free to check the logs. I tested with the following dependency version

```
{'transformers': '4.39.3', 'accelerate': '0.29.3', 'deepspeed': '0.14.1', 'datasets': '2.18.0'}
```

When not using deepspeed, the test passes

```
{'transformers': '4.39.3', 'accelerate': '0.29.3', 'deepspeed': '0.14.1', 'datasets': '2.18.0'}
(ref_logprob-logprob).exp().mean()=tensor(1.0000, device='cuda:0')
```

When using deepspeed 2 or 3, the test fails


```
(ref_logprob-logprob).exp().mean()=tensor(1.1328, device='cuda:0', dtype=torch.bfloat16)
Traceback (most recent call last):
  File "/fsx/costa/zero3_min_repro/min_repro.py", line 157, in <module>
    torch.testing.assert_close(ref_logprob, logprob, rtol=1e-2, atol=1e-2) # a very generous tolerance
  File "/fsx/costa/zero3_min_repro/.venv/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1520, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 2603 / 3200 (81.3%)
Greatest absolute difference: 14.5 at index (53, 15) (up to 0.01 allowed)
Greatest relative difference: 2416.0 at index (21, 28) (up to 0.01 allowed)
```

