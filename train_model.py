# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load dataset
# df = pd.read_csv("heart.csv")

# # Check dataset structure
# print("Dataset Loaded Successfully!")
# print(df.head())

# # Convert categorical columns to numeric using Label Encoding
# categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# for col in categorical_columns:
#     encoder = LabelEncoder()
#     df[col] = encoder.fit_transform(df[col])

# # Verify all numerical columns are included
# numerical_columns = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# # Ensure all required columns are present
# required_columns = numerical_columns + categorical_columns + ["HeartDisease"]
# missing_cols = [col for col in required_columns if col not in df.columns]
# if missing_cols:
#     raise ValueError(f"Missing columns in dataset: {missing_cols}")

# # Split data into features (X) and labels (y)
# X = df.drop(columns=["HeartDisease"])  # Ensure 'HeartDisease' is the correct column name
# y = df["HeartDisease"]

# # Standardize the numerical feature values
# scaler = StandardScaler()
# X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# # Reshape X to fit CNN input (samples, time steps, features)
# X_scaled = X.values.reshape(X.shape[0], X.shape[1], 1)

# # Split dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Define the CNN-based model
# model = Sequential([
#     Input(shape=(X_train.shape[1], 1)),  # Input layer
#     Conv1D(filters=32, kernel_size=3, activation='relu'),
#     Conv1D(filters=64, kernel_size=3, activation='relu'),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')  # Output layer for binary classification
# ])

# # Compile the model
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Train the model
# print("Training the model...")
# history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# # Save the trained model
# model.save("my_model.h5")
# print("✅ Model trained and saved as 'my_model.h5'")
