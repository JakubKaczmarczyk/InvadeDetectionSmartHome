import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

def prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Count samples
    total_samples = len(df)
    attack_samples = df[df['attack'] == 'Yes'].shape[0]
    normal_samples = df[df['attack'] == 'No'].shape[0]
    
    print(f"Total samples: {total_samples}")
    print(f"Attack samples: {attack_samples} ({attack_samples/total_samples*100:.2f}%)")
    print(f"Normal samples: {normal_samples} ({normal_samples/total_samples*100:.2f}%)")
    
    # Encode categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        # Zapisz encoder do pliku
        with open(f'{col}_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
    
    # Encode target
    le = LabelEncoder()
    df['attack'] = le.fit_transform(df['attack'])
    
    # Select features
    features = [
        'duration', 'protocol_type', 'service', 'flag', 
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
        'urgent', 'hot', 'logged_in', 'num_compromised', 
        'count', 'srv_count', 'serror_rate', 'rerror_rate', 
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
        'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate'
    ]
    
    X = df[features].values
    y = df['attack'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # One-hot encode labels
    y = to_categorical(y)
    
    return X, y, scaler

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_anomaly_detector(file_path, model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    # Prepare data
    X, y, scaler = prepare_data(file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = create_lstm_model(
        input_shape=(1, X.shape[2]), 
        num_classes=y.shape[1]
    )
    
    # Train model
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

     # Zapisywanie modelu LSTM
    print(f"Zapisuję model do pliku {model_path}...")
    model.save(model_path)
    print(f"Model zapisany jako {model_path}")
    
    # Zapisywanie skalera
    print(f"Zapisuję skaler do pliku {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Skaler zapisany jako {scaler_path}")
    
    return model, scaler

# Example usage
if __name__ == '__main__':
    model, scaler = train_anomaly_detector('dataset_invade.csv')