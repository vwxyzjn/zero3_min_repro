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

