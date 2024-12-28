# Potion

Live video inference platform which relies on VALI.
Use it to run your inference on network stream:
- RTSP
- HLS
- Or something else that FFMpeg supports

Usage example:
```
python 3 ./Potion.py \
  -i http://media.sever/rtsp_stream
  -o detections.json
  -t 5
  -d True
  -gpu_id 0
```

For example it runs SSD3 PyTorch model.