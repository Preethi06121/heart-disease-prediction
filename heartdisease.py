import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("framingham.csv")
data = data.dropna()  
x=data.drop('TenYearCHD',axis=1)
y=data['TenYearCHD']
 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

predictions=model.predict(x_train)
print("Accuracy:",accuracy_score(y_train,predictions))

print("The person has heart disease (yes=1/no=0)",model.predict([[1,44,2,0,18,0,0,1,0,222,144,77,44,78,75]]))
