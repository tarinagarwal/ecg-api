"""
ECG API Test Script
Tests the ECG analysis API with synthetic and sample ECG data
"""

import numpy as np
import httpx
import asyncio
import json
import time


# ============ Synthetic ECG Generator ============

def generate_synthetic_ecg(
    duration_seconds: float = 5.0,
    sampling_rate: int = 250,
    heart_rate: int = 72
) -> list:
    """
    Generate synthetic ECG signal with realistic PQRST morphology
    
    Args:
        duration_seconds: Duration of the signal
        sampling_rate: Sampling frequency in Hz
        heart_rate: Heart rate in BPM
        
    Returns:
        List of ECG values
    """
    num_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Beat period
    beat_period = 60.0 / heart_rate
    
    ecg = np.zeros(num_samples)
    
    # Generate each heartbeat
    beat_time = 0
    while beat_time < duration_seconds:
        # Time relative to this beat
        t_rel = t - beat_time
        
        # P wave (small positive deflection before QRS)
        p_center = 0.0
        ecg += 0.15 * np.exp(-((t_rel - p_center) ** 2) / (2 * 0.01 ** 2))
        
        # Q wave (small negative before R)
        q_center = 0.12
        ecg -= 0.1 * np.exp(-((t_rel - q_center) ** 2) / (2 * 0.005 ** 2))
        
        # R wave (tall positive spike)
        r_center = 0.15
        ecg += 1.0 * np.exp(-((t_rel - r_center) ** 2) / (2 * 0.008 ** 2))
        
        # S wave (negative after R)
        s_center = 0.18
        ecg -= 0.2 * np.exp(-((t_rel - s_center) ** 2) / (2 * 0.008 ** 2))
        
        # T wave (positive deflection after S)
        t_center = 0.35
        ecg += 0.3 * np.exp(-((t_rel - t_center) ** 2) / (2 * 0.03 ** 2))
        
        beat_time += beat_period
    
    # Add some noise
    ecg += np.random.normal(0, 0.02, num_samples)
    
    return ecg.tolist()


def generate_bradycardia_ecg(duration_seconds: float = 5.0) -> list:
    """Generate ECG with slow heart rate (bradycardia)"""
    return generate_synthetic_ecg(duration_seconds, heart_rate=45)


def generate_tachycardia_ecg(duration_seconds: float = 5.0) -> list:
    """Generate ECG with fast heart rate (tachycardia)"""
    return generate_synthetic_ecg(duration_seconds, heart_rate=120)


def generate_irregular_ecg(duration_seconds: float = 5.0) -> list:
    """Generate ECG with irregular rhythm"""
    sampling_rate = 250
    num_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    ecg = np.zeros(num_samples)
    
    # Variable beat intervals (irregular)
    beat_times = [0]
    current_time = 0
    while current_time < duration_seconds:
        # Random interval between 0.5 and 1.2 seconds
        interval = np.random.uniform(0.5, 1.2)
        current_time += interval
        if current_time < duration_seconds:
            beat_times.append(current_time)
    
    for beat_time in beat_times:
        t_rel = t - beat_time
        
        # P wave
        ecg += 0.15 * np.exp(-((t_rel - 0.0) ** 2) / (2 * 0.01 ** 2))
        # Q wave
        ecg -= 0.1 * np.exp(-((t_rel - 0.12) ** 2) / (2 * 0.005 ** 2))
        # R wave
        ecg += 1.0 * np.exp(-((t_rel - 0.15) ** 2) / (2 * 0.008 ** 2))
        # S wave
        ecg -= 0.2 * np.exp(-((t_rel - 0.18) ** 2) / (2 * 0.008 ** 2))
        # T wave
        ecg += 0.3 * np.exp(-((t_rel - 0.35) ** 2) / (2 * 0.03 ** 2))
    
    ecg += np.random.normal(0, 0.02, num_samples)
    return ecg.tolist()


# ============ Test Functions ============

def test_api_sync(base_url: str = "http://localhost:8000"):
    """Test the API synchronously"""
    print("\n" + "=" * 60)
    print("ECG API Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n[Test 1] Health Check...")
    try:
        response = httpx.get(f"{base_url}/health", timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        assert response.status_code == 200
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return
    
    # Test 2: Normal ECG
    print("\n[Test 2] Normal ECG Analysis...")
    ecg_normal = generate_synthetic_ecg(duration_seconds=5.0, heart_rate=72)
    try:
        response = httpx.post(
            f"{base_url}/analyze",
            json={"ecg": ecg_normal, "sampling_rate": 250},
            timeout=30
        )
        print(f"  Status: {response.status_code}")
        result = response.json()
        print(f"  Heart Rate: {result['heart_rate']} BPM")
        print(f"  R-peaks found: {len(result['r_peaks'])}")
        print(f"  Abnormalities: {result['abnormalities']}")
        print(f"  Diagnosis: {result['final_diagnosis'][:80]}...")
        assert response.status_code == 200
        assert 60 <= result['heart_rate'] <= 85
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Bradycardia
    print("\n[Test 3] Bradycardia ECG Analysis...")
    ecg_brady = generate_bradycardia_ecg()
    try:
        response = httpx.post(
            f"{base_url}/analyze",
            json={"ecg": ecg_brady, "sampling_rate": 250},
            timeout=30
        )
        result = response.json()
        print(f"  Heart Rate: {result['heart_rate']} BPM")
        print(f"  Abnormalities: {result['abnormalities']}")
        assert response.status_code == 200
        assert result['heart_rate'] < 60
        assert any("Bradycardia" in a for a in result['abnormalities'])
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 4: Tachycardia
    print("\n[Test 4] Tachycardia ECG Analysis...")
    ecg_tachy = generate_tachycardia_ecg()
    try:
        response = httpx.post(
            f"{base_url}/analyze",
            json={"ecg": ecg_tachy, "sampling_rate": 250},
            timeout=30
        )
        result = response.json()
        print(f"  Heart Rate: {result['heart_rate']} BPM")
        print(f"  Abnormalities: {result['abnormalities']}")
        assert response.status_code == 200
        assert result['heart_rate'] > 100
        assert any("Tachycardia" in a for a in result['abnormalities'])
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 5: Irregular rhythm
    print("\n[Test 5] Irregular Rhythm Analysis...")
    ecg_irregular = generate_irregular_ecg()
    try:
        response = httpx.post(
            f"{base_url}/analyze",
            json={"ecg": ecg_irregular, "sampling_rate": 250},
            timeout=30
        )
        result = response.json()
        print(f"  Heart Rate: {result['heart_rate']} BPM")
        print(f"  Abnormalities: {result['abnormalities']}")
        assert response.status_code == 200
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 6: Error handling - too short
    print("\n[Test 6] Error Handling (Too Short Signal)...")
    try:
        response = httpx.post(
            f"{base_url}/analyze",
            json={"ecg": [0.1, 0.2], "sampling_rate": 250},
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        assert response.status_code == 400
        print("  ✓ PASSED (correctly rejected)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)


async def test_websocket(base_url: str = "ws://localhost:8000"):
    """Test WebSocket connection"""
    import websockets
    
    print("\n[WebSocket Test] Connecting...")
    
    try:
        async with websockets.connect(f"{base_url}/ws") as ws:
            print("  Connected!")
            
            # Send ECG data
            ecg_data = generate_synthetic_ecg(duration_seconds=3.0)
            await ws.send(json.dumps({
                "ecg": ecg_data,
                "sampling_rate": 250
            }))
            
            # Receive response
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            result = json.loads(response)
            
            print(f"  Received analysis via WebSocket")
            print(f"  Heart Rate: {result['data']['heart_rate']} BPM")
            print("  ✓ WebSocket PASSED")
            
    except Exception as e:
        print(f"  ✗ WebSocket FAILED: {e}")


def simulate_realtime_stream(base_url: str = "http://localhost:8000", num_readings: int = 5):
    """Simulate real-time ECG readings from hardware"""
    print("\n" + "=" * 60)
    print("Simulating Real-time ECG Stream")
    print("=" * 60)
    
    for i in range(num_readings):
        print(f"\n[Reading {i+1}/{num_readings}]")
        
        # Generate ECG with slight heart rate variation
        heart_rate = 70 + np.random.randint(-10, 10)
        ecg = generate_synthetic_ecg(duration_seconds=3.0, heart_rate=heart_rate)
        
        try:
            start_time = time.time()
            response = httpx.post(
                f"{base_url}/analyze",
                json={"ecg": ecg, "sampling_rate": 250},
                timeout=30
            )
            elapsed = time.time() - start_time
            
            result = response.json()
            print(f"  HR: {result['heart_rate']} BPM | "
                  f"Beats: {len(result['r_peaks'])} | "
                  f"Status: {result['abnormalities'][0]} | "
                  f"Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(1)  # Simulate interval between readings
    
    print("\n" + "=" * 60)
    print("Stream Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stream":
            simulate_realtime_stream(base_url)
        elif sys.argv[1] == "--websocket":
            asyncio.run(test_websocket("ws://localhost:8000"))
        else:
            test_api_sync(base_url)
    else:
        test_api_sync(base_url)
