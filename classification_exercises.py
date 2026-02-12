import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification, load_iris

print("### Exercise 0: Environment and libraries ###")
print("Libraries imported successfully.\n")

print("### Exercise 1: Logistic regression in Scikit-learn ###")
X = [[0],[0.1],[0.2], [1],[1.1],[1.2], [1.3]]
y = [0,0,0,1,1,1,0]

# Fit the logistic regression
model = LogisticRegression()
model.fit(X, y)

# Predict the class for x_pred = [[0.5]]
x_pred = [[0.5]]
prediction = model.predict(x_pred)
print(f"Prediction for {x_pred}: {prediction[0]}")

# Predict the probabilities for x_pred = [[0.5]]
probabilities = model.predict_proba(x_pred)
print(f"Probabilities for {x_pred}: {probabilities}")

# Print coefficients, intercept, and score
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Score: {model.score(X, y)}\n")

print("### Exercise 2: Sigmoid ###")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid1(x):
    return 1 / (1 + np.exp(-(0.5*x + 3)))

def sigmoid2(x):
    return 1 / (1 + np.exp(-(5*x + 11)))

x = np.linspace(-10, 10, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label='sigmoid')
plt.plot(x, sigmoid1(x), label='sigmoid1')
plt.plot(x, sigmoid2(x), label='sigmoid2')
plt.axhline(y=0.5, color='r', linestyle='--', label='prob=0.5')
plt.legend()
plt.grid(True)
plt.title("Exercise 2: Sigmoid Functions")
# plt.show() # Commented out to prevent blocking in non-interactive environments, uncomment to view
print("Sigmoid plot created.\n")

print("### Exercise 3: Decision boundary ###")
X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.5, 0.5],
    flip_y=0.15,
    class_sep=2.0,
    hypercube=True,
    shift=1.0,
    scale=1.0,
    shuffle=True,
    random_state=88
)

model = LogisticRegression()
model.fit(X, y)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

def predict_probability(coefs, X_val):
    return 1 / (1 + np.exp(-(coefs[0] * X_val + coefs[1])))

coefs = [model.coef_[0][0], model.intercept_[0]]
sample_x = X[0][0]
prob_custom = predict_probability(coefs, sample_x)
prob_sklearn = model.predict_proba([[sample_x]])[0][1]
print(f"Custom Prob: {prob_custom}, Sklearn Prob: {prob_sklearn}")

def predict_class(coefs, X_val, threshold=0.5):
    prob = predict_probability(coefs, X_val)
    return 1 if prob >= threshold else 0

class_custom = predict_class(coefs, sample_x)
class_sklearn = model.predict([[sample_x]])[0]
print(f"Custom Class: {class_custom}, Sklearn Class: {class_sklearn}\n")


print("### Exercise 4: Train test split ###")
X = np.arange(1,21).reshape(10,-1)
y = np.zeros(10)
y[7:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"X_train: {X_train}")
print(f"y_train: {y_train}")
print(f"X_test: {X_test}")
print(f"y_test: {y_test}")
print(f"Train class 1 prop: {np.mean(y_train)}")
print(f"Test class 1 prop: {np.mean(y_test)}")

# Stratified split
X = np.arange(1,201).reshape(100,-1)
y = np.zeros(100)
y[70:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=43)
print(f"Stratified Train class 1 prop: {np.mean(y_train)}")
print(f"Stratified Test class 1 prop: {np.mean(y_test)}\n")


print("### Exercise 5: Breast Cancer prediction ###")
try:
    columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
               'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
               'Normal Nucleoli', 'Mitoses', 'Class']
    df = pd.read_csv('breast-cancer-wisconsin.data', names=columns, na_values='?')

    # Handle missing values
    df = df.filln(df.median())

    # Drop Sample code number
    df = df.drop('Sample code number', axis=1)

    # Proportion of Benign (Class 2)
    prop_benign = (df['Class'] == 2).mean()
    print(f"Proportion of Benign: {prop_benign}")

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=43)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print("Confusion Matrix Train:")
    print(confusion_matrix(y_train, y_pred_train))
    print("Confusion Matrix Test:")
    print(confusion_matrix(y_test, y_pred_test))
    print()
except Exception as e:
    print(f"Could not complete Exercise 5: {e}\n")


print("### Exercise 6: Multi-class (Optional) ###")
iris = load_iris()
X = pd.DataFrame(data=iris['data'], columns=iris.feature_names)
y = pd.Series(data=iris['target'], name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=43)

def train(X_train, y_train):
    try:
        clf0 = LogisticRegression(solver='liblinear')
        y0 = (y_train == 0).astype(int)
        clf0.fit(X_train, y0)

        clf1 = LogisticRegression(solver='liblinear')
        y1 = (y_train == 1).astype(int)
        clf1.fit(X_train, y1)

        clf2 = LogisticRegression(solver='liblinear')
        y2 = (y_train == 2).astype(int)
        clf2.fit(X_train, y2)

        return clf0, clf1, clf2
    except Exception as e:
        print(f"Error in train function: {e}")
        return None, None, None

clf0, clf1, clf2 = train(X_train, y_train)

if clf0 and clf1 and clf2:
    def predict_one_vs_all(X, clf0, clf1, clf2):
        prob0 = clf0.predict_proba(X)[:, 1]
        prob1 = clf1.predict_proba(X)[:, 1]
        prob2 = clf2.predict_proba(X)[:, 1]
        return np.argmax(np.column_stack([prob0, prob1, prob2]), axis=1)

    y_pred = predict_one_vs_all(X_test, clf0, clf1, clf2)
    print(f"Custom Accuracy: {accuracy_score(y_test, y_pred)}")

    clf_multi = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf_multi.fit(X_train, y_train)
    print(f"Sklearn Accuracy: {clf_multi.score(X_test, y_test)}")
else:
    print("Skipping Multi-class prediction due to training error.")
