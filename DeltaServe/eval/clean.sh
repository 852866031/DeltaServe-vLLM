ps aux | grep 'jiaxuan' | grep 'slora' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'jiaxuan' | grep 'auto_benchmark.py ' | grep -v grep | awk '{print $2}' | xargs -r kill -9