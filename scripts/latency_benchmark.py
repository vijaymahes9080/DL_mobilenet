import time
import numpy as np
import tensorflow as tf
from pathlib import Path

def benchmark_tflite(model_path, duration=60):
    print(f"🚀 Benchmarking {model_path} for {duration} seconds...")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    start_time = time.time()
    latencies = []
    frames = 0
    
    while (time.time() - start_time) < duration:
        input_data = np.random.random_sample(input_shape).astype(np.float32)
        
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000)
        frames += 1
        if frames % 100 == 0:
            print(f"  Processed {frames} frames... Current Avg: {np.mean(latencies):.2f}ms")
            
    avg_latency = np.mean(latencies)
    print(f"\n✅ BENCHMARK COMPLETE")
    print(f"  Total Frames: {frames}")
    print(f"  Average Latency: {avg_latency:.2f}ms")
    print(f"  P95 Latency: {np.percentile(latencies, 95):.2f}ms")
    
    return avg_latency

if __name__ == "__main__":
    model = Path("models/autonomous/optimized_model_fp16.tflite")
    if model.exists():
        avg = benchmark_tflite(model)
        if avg >= 50:
            print("⚠️ LATENCY THRESHOLD VIOLATED (>= 50ms).")
            print("🛑 TRIGGERING INT8 QUANTIZATION...")
        else:
            print("💎 LATENCY REQUIREMENT MET (< 50ms).")
    else:
        print(f"❌ Error: Model {model} not found.")
