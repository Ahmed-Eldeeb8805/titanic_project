import numpy as np
import pandas as pd
import json

df=pd.read_csv('train.csv')
Y=df['Survived'].values
df=df.drop( columns=['PassengerId','Name','Ticket','Cabin','Survived'])
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].map({
    'S': 0,
    'C': 1,
    'Q': 2
})
df['Embarked']=df['Embarked'].fillna(0)
df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})

print("columns order:", list(df.columns))

X=df.values

mean = X.mean(axis=0)
std = X.std(axis=0)

X = (X - mean) / std

def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g

def compute_cost(x,y,w,b):
    cost=0
    m=len(x)
    for i in range(m):
        f_x=sigmoid((np.dot(x[i],w))+b)
        cost+=(y[i]*np.log(f_x))+((1-y[i])*np.log(1-f_x))
    cost=cost/-m
    return cost

def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range (m):
        f_x=sigmoid((np.dot(x[i],w))+b)
        err= f_x-y[i]
        dj_db+=err
        for j in range (n):
            dj_dw[j]+=err*x[i,j]
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

def gradient(x,y,w_in,b_in,alpha,iterations):
    j_hist=[]
    w=w_in.copy()
    b=b_in
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = compute_cost(x, y, w, b)
        j_hist.append(cost)
        if i % 100 == 0 or i == iterations - 1:
            print(f"Iteration {i:4d}: Cost {cost}")
    return w, b, j_hist

def prediction(x,w,b):
    predict=sigmoid(np.dot(x,w)+b)
    return (predict >= 0.5).astype(int)

def train_test_split(X, Y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    m = X.shape[0]
    indices = np.random.permutation(m)
    test_size = int(m * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_ratio=0.2)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

n = X_train.shape[1]
w_init = np.zeros(n)
b_init = 0.0
alpha = 0.1
num_iters = 3000

w_final, b_final, J_history = gradient(X_train, Y_train, w_init, b_init, alpha, num_iters)

train_acc = np.mean(prediction(X_train, w_final, b_final) == Y_train)
test_acc = np.mean(prediction(X_test, w_final, b_final) == Y_test)
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# تصدير كل القيم اللازمة للفرونت اند
export = {
    "columns": list(df.columns),
    "w": w_final.tolist(),
    "b": float(b_final),
    "mean": mean.tolist(),
    "std": std.tolist(),
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "cost_history": J_history
}

with open("model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\n--- Exported values ---")
print("columns:", export["columns"])
print("w:", export["w"])
print("b:", export["b"])
print("mean:", export["mean"])
print("std:", export["std"])
