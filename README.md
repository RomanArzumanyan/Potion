# Potion

Live video inference tool which relies on VALI.
Use it to run your inference on network stream:
- RTSP
- HLS
- Or something else that FFMpeg supports

Install:
```
pip install python_potion
```

Usage example:
```
python3 -m python_potion \
  -i http://media.sever/rtsp_stream
  -o detections.json
  -t 5
  -d dump
  -n 3
  -gpu_id 0
```

What does it mean ?
```
options:
  -h, --help            show this help message and exit
  -gpu_id GPU_ID, --gpu-id GPU_ID
                        GPU id, check nvidia-smi
  -i INPUT, --input INPUT
                        Encoded video file (read from)
  -o OUTPUT, --output OUTPUT
                        output json file name
  -t TIME, --time TIME  processing time, s.
  -d DUMP, --dump DUMP  dump video filename without extension
  -n NUM_RETRIES, --num_retries NUM_RETRIES
                        number of attepts to respawn video reader in case of failure
```

For example it runs SSD3 PyTorch model.