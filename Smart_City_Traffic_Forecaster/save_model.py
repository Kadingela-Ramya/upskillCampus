# save_model.py
# Run this at the END of your Colab notebook to save the trained model and scalers
# Then download lstm_model.h5, scaler_X.pkl, scaler_y.pkl and place in flask_app folder

import joblib

# Save LSTM model (assuming your model variable is called 'model')
model.save('lstm_model.h5')
print("Saved lstm_model.h5")

# Save scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Saved scaler_X.pkl and scaler_y.pkl")

print("\nNow download these 3 files from Colab and place them in your flask_app folder!")
