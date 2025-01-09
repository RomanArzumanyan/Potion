# Potion

Live video inference tool which relies on VALI.
Use it to run the inference on local and network streams.
It uses double queue (compressed + reconstructed) in order not to miss a single frame if inference is slow.

As for early version (0.2 for now) it is only tested with densenet.

Install:
```
pip install python_potion
```

Usage example:
```
python3 -m python_potion \
  -i http://media.sever/rtsp_stream
  "-t",
  "10",
  "-gpu_id",
  "0",
  "-d",
  "dump",
  "-m",
  "densenet_onnx",
  "-x",
  "12",
  "-u",
  "0.0.0.0:8001"
```

What does it mean ?
```
options:
  -h, --help            show this help message and exit
  -gpu_id GPU_ID, --gpu-id GPU_ID
                        GPU id, check nvidia-smi
  -i INPUT, --input INPUT
                        Encoded video file (read from)
  -t TIME, --time TIME  processing time, s.
  -d DUMP, --dump DUMP  dump video filename without extension
  -n NUM_RETRIES, --num_retries NUM_RETRIES
                        number of attepts to respawn video reader in case of failure
  -v, --verbose         Enable verbose output
  -m MODEL_NAME, --model-name MODEL_NAME
                        Name of model
  -x MODEL_VERSION, --model-version MODEL_VERSION
                        Version of model. Default is to use latest version.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Default is 1.
  -c CLASSES, --classes CLASSES
                        Number of class results to report. Default is 1.
  -s {NONE,INCEPTION,VGG}, --scaling {NONE,INCEPTION,VGG}
                        Type of scaling to apply to image pixels. For now it is NONE.
  -u URL, --url URL     Inference server URL. Default is localhost:8000.
  -p PROTOCOL, --protocol PROTOCOL
                        Protocol used to communicate with the inference service. For now it is gRPC only.
```

For example it runs SSD3 PyTorch model.