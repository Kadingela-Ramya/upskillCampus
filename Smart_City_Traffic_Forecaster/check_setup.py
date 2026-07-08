# check_setup.py
# Run this to diagnose your environment before starting Flask
# Usage: python check_setup.py

import sys
print(f"Python: {sys.version}")
print()

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    # Try loading keras
    try:
        model_test = tf.keras.Sequential()
        print("✅ tf.keras works fine")
    except Exception as e:
        print(f"⚠️  tf.keras issue: {e}")
        print("   → Try: pip install tf-keras")
except ImportError:
    print("❌ TensorFlow NOT installed")
    print("   → Run: pip install tensorflow==2.15.0")
except Exception as e:
    print(f"❌ TensorFlow error: {e}")

print()

# Check joblib
try:
    import joblib
    print(f"✅ joblib: {joblib.__version__}")
except ImportError:
    print("❌ joblib NOT installed → Run: pip install joblib")

print()

# Check model files
import os
files = ['lstm_model.h5', 'scaler_X.pkl', 'scaler_y.pkl']
all_found = True
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"✅ {f} found ({size:,} bytes)")
    else:
        print(f"❌ {f} NOT FOUND — paste this file next to app.py")
        all_found = False

print()
if all_found:
    print("🚀 All files present! Run: python app.py")
    print("   The dashboard will use REAL predictions.")
else:
    print("⚠️  Missing model files — Flask will run in DEMO mode.")
    print("   Download them from Colab and paste into this folder.")
