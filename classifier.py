import serial
import time
import numpy as np

# Configuration
PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
CALIBRATION_TIME = 5  # Seconds to calibrate resting state

def run_adaptive_classifier():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to {PORT}.")
        
        # --- CALIBRATION PHASE ---
        print(f"--- CALIBRATION: Keep muscle RELAXED for {CALIBRATION_TIME}s ---")
        baseline_data = []
        start_cal = time.time()
        
        while time.time() - start_cal < CALIBRATION_TIME:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) == 2:
                    baseline_data.append(float(parts[1]))
        
        if not baseline_data:
            print("No data received during calibration. Check connection.")
            return

        mean_rest = np.mean(baseline_data)
        std_rest = np.std(baseline_data)
        # Threshold = mean resting value + 5 standard deviations to avoid noise
        threshold = mean_rest + (std_rest * 5)
        
        print(f"Calibration Complete!")
        print(f"Resting Mean: {mean_rest:.2f} | Threshold Set To: {threshold:.2f}")
        print("--- Starting Detection ---")

        # --- DETECTION PHASE ---
        active_start_time = None
        triggered_1s = False
        triggered_3s = False

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) < 2: continue
                
                envelope = float(parts[1])

                if envelope >= threshold:
                    if active_start_time is None:
                        active_start_time = time.time()
                    
                    duration = time.time() - active_start_time

                    if duration >= 1.0 and not triggered_1s:
                        print("\n[EVENT] 1s: YES")
                        triggered_1s = True
                    
                    if duration >= 3.0 and not triggered_3s:
                        print("\n[EVENT] 3s: NO")
                        triggered_3s = True

                    print(f"Holding: {duration:.2f}s | Env: {envelope:3.0f}", end='\r')
                else:
                    active_start_time = None
                    triggered_1s = False
                    triggered_3s = False
                    print(f"Resting... (Threshold: {threshold:.1f})", end='\r')

    except serial.SerialException as e:
        print(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    run_adaptive_classifier()