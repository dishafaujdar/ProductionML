import pickle

def save_model(model, scaler, feature_names, filepath='model.pkl'):
    """
    Save trained model, scaler, and feature names.
    
    Args:
        model: Trained MyLinearRegression object
        scaler: Fitted StandardScaler object
        feature_names: List of feature names in correct order
        filepath: Path to save the model
    """
    artifact = {
        'weights': model.weights,
        'scaler': scaler,
        'feature_names': feature_names  # Pass actual feature names
    }
    with open(filepath, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {filepath}")

def load_model(filepath='model.pkl'):
    """
    Load saved model, scaler, and feature names.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        weights: Model weights (numpy array)
        scaler: StandardScaler object
        feature_names: List of feature names
    """
    with open(filepath, 'rb') as f:
        artifact = pickle.load(f)
    return artifact['weights'], artifact['scaler'], artifact['feature_names']