import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load data
data_path = os.path.join('data', 'YearPredictionMSD.txt')
column_names = ['Year'] + [f'feature_{i}' for i in range(1, 91)]
data = pd.read_csv(data_path, header=None, names=column_names)

# Map years to decades
def year_to_decade(year):
    decade = int(year / 10) * 10
    return decade

data['Decade'] = data['Year'].apply(year_to_decade)
X = data.drop(['Year', 'Decade'], axis=1)
y = data['Decade']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("Preprocessing done. Files saved in 'data/' folder.")
