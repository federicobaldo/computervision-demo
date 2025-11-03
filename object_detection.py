import asyncio
import logging
import timeit
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
          integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" 
          crossorigin="anonymous">
    <title>Webcam Live Streaming</title>
</head>
<body>
<div class="container">
    <div class="row">
        <div class="col-lg-8 offset-lg-2">
            <h3 class="mt-5">Webcam Live Streaming</h3>
            <img src="/video_feed" width="100%" height="auto" class="img-fluid" alt="Live Video Feed">
        </div>
    </div>
</div>
</body>
</html>
"""

# COCO dataset labels (91 classes)
LABEL_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "none", 12: "stop sign", 13: "parking meter",
    14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    25: "none", 26: "backpack", 27: "umbrella", 28: "none", 29: "none",
    30: "handbag", 31: "tie", 32: "suitcase", 33: "frisbee", 34: "skis",
    35: "snowboard", 36: "sports ball", 37: "kite", 38: "baseball bat",
    39: "baseball glove", 40: "skateboard", 41: "surfboard", 42: "tennis racket",
    43: "bottle", 44: "none", 45: "wine glass", 46: "cup", 47: "fork",
    48: "knife", 49: "spoon", 50: "bowl", 51: "banana", 52: "apple",
    53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot", 57: "hot dog",
    58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch",
    63: "potted plant", 64: "bed", 65: "none", 66: "dining table",
    67: "none", 68: "none", 69: "toilet", 70: "none", 71: "tv",
    72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard", 76: "cell phone",
    77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator",
    82: "none", 83: "book", 84: "clock", 85: "vase", 86: "scissors",
    87: "teddy bear", 88: "hair drier", 89: "toothbrush"
}

# Global configuration
CAMERA_INDEX = 4
CONFIDENCE_THRESHOLD = 0.5
MAX_WORKERS = 3

# Thread pool executor for blocking operations
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# TFLite model globals
tflite_interpreter = None
tflite_input_details = None
tflite_output_details = None
tflite_input_shape = None


def init_tflite_model(model_path, delegate_path=None):
    """
    Initialize TensorFlow Lite interpreter with optional hardware delegate.

    Args:
        model_path: Path to the .tflite model file
        delegate_path: Optional path to external delegate library (e.g., NPU, GPU)

    Raises:
        Exception: If model initialization fails
    """
    global tflite_interpreter, tflite_input_details, tflite_output_details, tflite_input_shape

    try:
        # Load interpreter with or without delegate
        if delegate_path:
            logger.info(f"Loading TFLite model with delegate: {delegate_path}")
            ext_delegate = [tflite.load_delegate(delegate_path)]
            tflite_interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=ext_delegate
            )
        else:
            logger.info("Loading TFLite model in CPU-only mode")
            tflite_interpreter = tflite.Interpreter(model_path=model_path)

        # Allocate tensors
        tflite_interpreter.allocate_tensors()
        
        # Get input/output details
        tflite_input_details = tflite_interpreter.get_input_details()
        tflite_output_details = tflite_interpreter.get_output_details()
        tflite_input_shape = tflite_input_details[0]['shape']

        # Log model information
        logger.info(f"Model input shape: {tflite_input_shape}")
        logger.info(f"Number of outputs: {len(tflite_output_details)}")
        
        for i, output in enumerate(tflite_output_details):
            logger.info(
                f"Output {i}: shape={output['shape']}, "
                f"dtype={output['dtype']}, "
                f"name={output.get('name', 'unknown')}"
            )

    except Exception as e:
        logger.error(f"Failed to initialize TFLite model: {e}")
        raise


def detect_objects_tflite(frame, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Perform object detection on a frame using TensorFlow Lite model.

    Args:
        frame: Input frame as numpy array (BGR format)
        confidence_threshold: Minimum confidence score for detections (0.0-1.0)

    Returns:
        Annotated frame with bounding boxes and labels
    """
    if tflite_interpreter is None:
        logger.error("TFLite interpreter not initialized. Call init_tflite_model() first.")
        return frame

    try:
        # Prepare input
        height = tflite_input_shape[1]
        width = tflite_input_shape[2]
        img_resized = cv2.resize(frame, (width, height)).astype(np.uint8)
        input_data = np.expand_dims(img_resized, axis=0)

        # Run inference
        tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
        tflite_interpreter.invoke()

        # Parse outputs based on model format
        num_outputs = len(tflite_output_details)

        if num_outputs == 4:
            # Standard SSD format: [boxes, classes, scores, num_detections]
            boxes = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])[0]
            labels = tflite_interpreter.get_tensor(tflite_output_details[1]['index'])[0]
            scores = tflite_interpreter.get_tensor(tflite_output_details[2]['index'])[0]
            num_detections = tflite_interpreter.get_tensor(tflite_output_details[3]['index'])[0]
        elif num_outputs == 1:
            # Single output format (classification or other format)
            output = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])[0]
            logger.warning(f"Single output format detected with shape: {output.shape}")
            logger.warning("This appears to be a classification model, not an object detection model")
            return frame
        else:
            logger.error(f"Unexpected number of outputs: {num_outputs}")
            return frame

        # Draw detections on frame
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        for i in range(int(num_detections)):
            if scores[i] > confidence_threshold:
                # Parse bounding box (normalized coordinates: [ymin, xmin, ymax, xmax])
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Convert to pixel coordinates
                x0 = max(0, int(xmin * frame_width))
                y0 = max(0, int(ymin * frame_height))
                x1 = min(frame_width, int(xmax * frame_width))
                y1 = min(frame_height, int(ymax * frame_height))

                # Get label information
                label_id = int(labels[i])
                label_name = LABEL_MAP.get(label_id, f"class_{label_id}")
                confidence = scores[i]
                label_text = f"{label_name} {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                
                # Draw label background
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    annotated_frame, 
                    (x0, y0 - text_size[1] - 4), 
                    (x0 + text_size[0], y0), 
                    (255, 0, 0), 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame, 
                    label_text, 
                    (x0, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )

        return annotated_frame

    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        return frame


async def index(request):
    """Serve the main HTML page."""
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def video_feed(request):
    """
    Stream video feed with object detection.
    
    Returns:
        Multipart JPEG stream response
    """
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame'}
    )
    await response.prepare(request)

    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(executor, lambda: cv2.VideoCapture(CAMERA_INDEX))

    async def get_frame():
        """Capture a frame from the camera."""
        return await loop.run_in_executor(executor, cap.read)

    async def detect(frame):
        """Run object detection on a frame."""
        return await loop.run_in_executor(executor, detect_objects_tflite, frame)

    async def encode(frame):
        """Encode frame as JPEG."""
        return await loop.run_in_executor(executor, lambda: cv2.imencode('.jpg', frame)[1].tobytes())

    try:
        prev_time = timeit.default_timer()
        fps = 0.0

        while True:
            # Capture frame
            ret, frame = await get_frame()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # Detect objects
            annotated_frame = await detect(frame)
            
            # Calculate FPS
            current_time = timeit.default_timer()
            time_diff = current_time - prev_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_time = current_time

            # Draw FPS counter
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.2f}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )

            # Encode and stream frame
            jpg_bytes = await encode(annotated_frame)
            
            try:
                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
                )
            except (ConnectionResetError, asyncio.CancelledError):
                logger.info("Client disconnected")
                break

    finally:
        # Cleanup
        await loop.run_in_executor(executor, cap.release)
        try:
            await response.write_eof()
        except Exception:
            pass

    return response


def main():
    """Initialize and run the web application."""
    # Initialize TFLite model
    model_path = "/home/guest/ssd_mobilenet_v1.tflite"
    delegate_path = "/usr/lib/libvx_delegate.so"
    
    try:
        logger.info("Attempting to load model with hardware delegate...")
        init_tflite_model(model_path, delegate_path=delegate_path)
        logger.info("Successfully loaded model with hardware acceleration")
    except Exception as e:
        logger.warning(f"Failed to load with delegate: {e}")
        logger.info("Falling back to CPU-only mode...")
        try:
            init_tflite_model(model_path)
            logger.info("Successfully loaded model in CPU-only mode")
        except Exception as e2:
            logger.error(f"Failed to initialize model: {e2}")
            raise

    # Create and run web application
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/video_feed', video_feed)
    
    logger.info("Starting web server on http://0.0.0.0:8080")
    web.run_app(app, host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
