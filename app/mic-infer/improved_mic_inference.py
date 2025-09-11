import tensorflow as tf
import numpy as np
import sounddevice as sd
import time
from collections import deque, Counter

commands = np.array(['up', 'down', 'left', 'right'])

# Load model
interpreter = tf.lite.Interpreter(model_path="keyword_spotting_robust.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class SmartDetector:
    def __init__(self):
        self.background_command = None
        self.baseline_energy = 0.005
        
    def calibrate(self):
        """Learn what the background falsely triggers"""
        print("🔧 Learning your environment (5 seconds of silence please)...")
        
        detections = []
        for i in range(10):
            audio = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            
            waveform = np.squeeze(audio)
            energy = np.sqrt(np.mean(np.square(waveform)))
            
            if energy > 0.001:  # Not complete silence
                spectrogram = preprocess_audio(waveform)
                prediction, confidence, _ = predict_command(spectrogram)
                
                if confidence > 0.8:
                    detections.append(prediction)
                    print(f"  Background triggers: {prediction} ({confidence:.1%})")
        
        # Find the most common false positive
        if detections:
            most_common = Counter(detections).most_common(1)[0]
            if most_common[1] >= 3:
                self.background_command = most_common[0]
                print(f"\n⚠️  Your environment triggers '{self.background_command}' frequently")
                print(f"   Will require extra verification for this command\n")
        else:
            print("✅ Clean environment detected\n")

detector = SmartDetector()

def record_audio(duration=1, sample_rate=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def preprocess_audio(waveform):
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.9
    
    waveform = waveform[:16000]
    padding = max(0, 16000 - len(waveform))
    if padding > 0:
        waveform = np.pad(waveform, (0, padding), mode='constant')
    
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    spectrogram = tf.signal.stft(waveform_tf, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    
    return spectrogram.numpy()

def predict_command(spectrogram):
    interpreter.set_tensor(input_details[0]['index'], spectrogram)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probabilities = tf.nn.softmax(output[0]).numpy()
    
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    
    return commands[predicted_index], confidence, probabilities

def is_real_command(prediction, confidence, energy, probabilities):
    """Determine if this is a real command or background noise"""
    
    # If this is the problematic background command
    if prediction == detector.background_command:
        # Require VERY high confidence and energy
        if confidence > 0.98 and energy > 0.02:
            # Also check if other probabilities are very low
            others_sum = sum(p for i, p in enumerate(probabilities) if commands[i] != prediction)
            if others_sum < 0.02:  # Others total < 2%
                return True
        return False
    
    # For other commands, normal thresholds
    return confidence > 0.85 and energy > 0.008

def main():
    print("🚀 Keyword Spotting System")
    print(f"📝 Commands: {', '.join(commands)}")
    
    detector.calibrate()
    
    print("Ready! Speak commands clearly.\n")
    print("Tip: Speak louder/clearer if your command isn't detected\n")
    
    try:
        last_detection_time = 0
        silence_count = 0
        
        while True:
            waveform = record_audio()
            energy = np.sqrt(np.mean(np.square(waveform)))
            
            # Check for silence
            if energy < 0.003:
                silence_count += 1
                if silence_count >= 3:
                    print("🔇 Silence\n")
                    silence_count = 0
                time.sleep(0.1)
                continue
            
            silence_count = 0
            
            # Predict
            spectrogram = preprocess_audio(waveform)
            prediction, confidence, probabilities = predict_command(spectrogram)
            
            # Display
            prob_str = ", ".join([f"{cmd}: {prob:.1%}" for cmd, prob in zip(commands, probabilities)])
            print(f"Energy: {energy:.3f} | {prob_str}")
            
            # Check if it's real
            current_time = time.time()
            
            if is_real_command(prediction, confidence, energy, probabilities):
                if current_time - last_detection_time > 0.8:
                    print(f"✅ DETECTED: '{prediction}' ({confidence:.1%})\n")
                    last_detection_time = current_time
                else:
                    print("(Too soon after last detection)\n")
            else:
                if prediction == detector.background_command:
                    print(f"🔊 Background noise (sounds like '{prediction}')\n")
                elif confidence > 0.6:
                    print(f"❓ Uncertain - try speaking louder\n")
                else:
                    print("❌ No command\n")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n👋 Stopped")

if __name__ == "__main__":
    main()