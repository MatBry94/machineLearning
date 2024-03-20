import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ładowanie danych bezpośrednio z oryginalnego źródła
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Przekształcanie danych w DataFrame
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
dataset = pd.DataFrame(data, columns=column_names)
dataset['MEDV'] = target

# Podział danych na cechy i zmienną celu
X = dataset.drop('MEDV', axis=1)  # Cechy
y = dataset['MEDV']  # Zmienna celu

# Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu regresji liniowej
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Przewidywanie na danych testowych
y_pred = linear_model.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Wyświetlenie wyników oceny
print(f'MSE: {mse}')
print(f'R^2: {r2}')

# Histogramy cech
plt.figure(figsize=(20, 15))
for i, col in enumerate(dataset.columns):
    plt.subplot(5, 3, i + 1)
    sns.histplot(dataset[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Heatmapa korelacji
corr_matrix = dataset.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Heatmapa korelacji między cechami')
plt.show()

# Wykres rzeczywiste vs. przewidywane ceny domów
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Rzeczywiste ceny domów')
plt.ylabel('Przewidywane ceny domów')
plt.title('Rzeczywiste vs. Przewidywane ceny domów')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linia idealnych przewidywań
plt.show()