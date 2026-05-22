# DeltaServe

DeltaServe is built on and adapted from [S-LoRA](https://github.com/S-LoRA/S-LoRA)

```
conda create -n dserve python=3.9
conda activate dserve
# cuda > 12.6
pip install torch==2.8.0 triton==3.4.0
pip install -e . --no-build-isolation
```

To use the plotting scripts, you also need to install
```
pip install pandas matplotlib
```


## Llama3 Experiments

Generate two dummy LoRA adapters for experiments (trains and copies automatically):
```
cd eval/llama3
python init_adapters.py
```

Launch a server with llama3 model loaded on a single GPU:
```
cd eval/llama3
python launch_llama3.py
```

Auto benchmark that uses a timeline file to generate requests and feed to the server:
```
cd eval/llama3
# For co-serving
python auto_benchmark.py --co
# For inference only
python auto_benchmark.py
```

See [eval/llama3/README.md](eval/llama3/README.md) for the full experiment reference.
