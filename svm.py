import numpy as np
import pandas as pd
import time
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib
#parameters to adjust
model_name = "svc_model_1500.pkl"
train_size = 1500
FEATURE_COUNT = 4 # creates a 4 qubit system

#load and split data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

X = train_df.drop('label', axis=1)
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= int(train_size/4), train_size=train_size, random_state=100)
# --- Process the dataframes for initializing the Quantum Kernel ---

pca = PCA(n_components=FEATURE_COUNT)

pca.fit(X_train)
X_train_features = pca.transform(X_train) #creates 4 features for 4 qubits

X_test_features = pca.transform(X_test)

# --- create classical SVM pipeline ---
clf = make_pipeline(MinMaxScaler((0, 2 * np.pi)), SVC(gamma='auto'))

print("kernel initialized, now training")
start = time.time()

clf.fit(X_train_features, y_train) # train the classical svm

end = time.time()
train_time = end - start

joblib.dump(clf, model_name) # save the model
print("time to train: " + str(train_time))
print("finished training, now scoring")

qsvc_score = clf.score(X_test_features, y_test)

print(f"QSVC classification test score: {qsvc_score}")