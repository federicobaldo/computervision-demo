# computervision-demo
Repo to collect some demos running on Eurotech ReliaGATE 15A-14

## Object Detection
Make sure that your system has the following package installed:
```
libtensorflow-lite2.14.0-2.14.0-r0.cortexa53_crypto.rpm
libtensorflow-lite-dbg-2.14.0-r0.cortexa53_crypto.rpm
libtensorflow-lite-dev-2.14.0-r0.cortexa53_crypto.rpm
tensorflow-lite-vx-delegate-2.14.0-r0.cortexa53_crypto.rpm
tensorflow-lite-vx-delegate-dbg-2.14.0-r0.cortexa53_crypto.rpm
tensorflow-lite-vx-delegate-dev-2.14.0-r0.cortexa53_crypto.rpm
```

Create a python virtual env
`python3.10 -m venv .venv`

Install dependencies
`pip install -r requirements.txt`

Download the object detection model and add the correct path to object_detection.py
`wget  https://github.com/ARM-software/ML-zoo/raw/master/models/object_detection/ssd_mobilenet_v1/tflite_uint8/ssd_mobilenet_v1.tflite`

Verify the camera index on the device and update the CAMERA_INDEX variable accordingly. For instance you can connect a usb camera and execute `ls /dev/video*` and pick the latest device created by date.

Run the script
`python object_detection.py`
```
2025-11-03 18:17:05,489 [INFO] Attempting to load model with hardware delegate...
2025-11-03 18:17:05,491 [INFO] Loading TFLite model with delegate: /usr/lib/libvx_delegate.so
INFO: Vx delegate: allowed_cache_mode set to 0.
INFO: Vx delegate: device num set to 0.
INFO: Vx delegate: allowed_builtin_code set to 0.
INFO: Vx delegate: error_during_init set to 0.
INFO: Vx delegate: error_during_prepare set to 0.
INFO: Vx delegate: error_during_invoke set to 0.
WARNING: Fallback unsupported op 32 to TfLite
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
2025-11-03 18:17:05,513 [INFO] Model input shape: [  1 300 300   3]
2025-11-03 18:17:05,513 [INFO] Number of outputs: 4
2025-11-03 18:17:05,514 [INFO] Output 0: shape=[ 1 10  4], dtype=<class 'numpy.float32'>, name=TFLite_Detection_PostProcess
2025-11-03 18:17:05,515 [INFO] Output 1: shape=[ 1 10], dtype=<class 'numpy.float32'>, name=TFLite_Detection_PostProcess:1
2025-11-03 18:17:05,515 [INFO] Output 2: shape=[ 1 10], dtype=<class 'numpy.float32'>, name=TFLite_Detection_PostProcess:2
2025-11-03 18:17:05,516 [INFO] Output 3: shape=[1], dtype=<class 'numpy.float32'>, name=TFLite_Detection_PostProcess:3
2025-11-03 18:17:05,517 [INFO] Successfully loaded model with hardware acceleration
2025-11-03 18:17:05,518 [INFO] Starting web server on http://0.0.0.0:8080
======== Running on http://0.0.0.0:8080 ========
(Press CTRL+C to quit)
```

Point your browser at the device ip and port 8080
