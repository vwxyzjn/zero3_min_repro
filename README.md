# get started

```
pip install -r requirements.txt
```

# run the following to get the logs


```bash
python min_repro.py > logs/1gpu.log 2>&1
accelerate launch --num_processes 8  min_repro.py > logs/8gpu.log 2>&1
accelerate launch --config_file deepspeed3.yaml  min_repro.py --deepspeed3 > logs/deepspeed3.log 2>&1
accelerate launch --config_file deepspeed2.yaml  min_repro.py --deepspeed2 > logs/deepspeed2.log 2>&1

