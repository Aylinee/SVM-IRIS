import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Veri setini dosyadan yükleyin
iris = pd.read_csv('C:/Users/AylinF/Desktop/IRIS/Iris.csv')

# Display data
print(iris.head())

# Show classes
print(iris['Species'].unique())

# Data set varieties are 3
print(iris.describe(include='all'))
print(iris.info())

# Remove unneeded column
iris.drop(columns="Id", inplace=True)

# Check if anything is missing
print(iris.isnull().sum())

# Split data into features and target variable
X = iris.iloc[:, 0:4].values
y = iris.iloc[:, 4].values

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM sınıflandırıcısını oluşturun
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Test verisi üzerinde tahminler yapın
y_pred = svm.predict(X_test)

# Başarım metriklerini hesaplayın
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

accuracy_percentage = accuracy * 100
recall_percentage = recall * 100
precision_percentage = precision * 100

print(f"Doğruluk: {accuracy_percentage:.2f}%")
print(f"Duyarlılık: {recall_percentage:.2f}%")
print(f"Özgüllük: {precision_percentage:.2f}%")
