import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("/home/kit/Downloads/Untitled-spreadsheet-Sheet1 (1).csv")
# print(data)

ET = pd.DataFrame(data["Estimated task"])
Act = pd.DataFrame(data["Actual"])
# print(ET)
lm = linear_model.LinearRegression()
model = lm.fit(ET, Act)
print(model.coef_)
print(model.intercept_)
print("Equation: y = "+str(model.coef_[0][0]) + "x" + str(model.intercept_[0]) )

list_ET = [int(row) for row in ET.values]
list_Act = [int(row) for row in Act.values]

n = len(ET)
sum = 0
for i in range (0, n):
    diff = list_ET[i] - list_Act[i]
    sq_diff = diff**2
    sum = sum + sq_diff
MSE = sum/n
print("MSE = " + str(MSE))

data.plot(kind="scatter", x="Estimated task", y= "Actual")
plt.plot(ET, model.predict(ET))
plt.show()