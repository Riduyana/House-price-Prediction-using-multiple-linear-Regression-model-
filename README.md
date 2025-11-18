import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv("/content/drive/MyDrive/Machine Learning/house_price.csv")
df.head()
<img width="950" height="313" alt="image" src="https://github.com/user-attachments/assets/7133bbc5-7990-4cec-837f-a823ab845603" />


x= df[['beds','baths','size']]
y= df['price']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model= LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse= mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Square Error:", mse)
print("R2 Score: ", r2)

<img width="570" height="76" alt="image" src="https://github.com/user-attachments/assets/98629041-b409-4d46-87d2-e320368efcd4" />


plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()
<img width="1066" height="722" alt="image" src="https://github.com/user-attachments/assets/1bb21b11-6e21-4bea-81f8-659216a86db7" />


new_data=np.array([[2,2,1080]])
new_price=model.predict(new_data)
print("Predicted price of new data is: ", new_price)
<img width="735" height="47" alt="image" src="https://github.com/user-attachments/assets/cc7778cb-b6ce-45b8-9b6d-a7b90c251228" />






