from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

app = Flask(__name__)

# Train Models
data1 = {'qty': [10,20,30,40,50,60,70,80], 'sales': [50000,80000,110000,140000,170000,200000,230000,260000]}
df1 = pd.DataFrame(data1)
model1 = LinearRegression()
model1.fit(df1[['qty']], df1['sales'])

data2 = {'Age': [22,25,27,30,32,35,38,40,42,45,48,50,55,60], 'Buy': [0,0,0,0,1,0,1,1,1,1,1,1,1,1]}
df2 = pd.DataFrame(data2)
model2 = LogisticRegression()
model2.fit(df2[['Age']], df2['Buy'])

HTML = """
<h1>Multan Shop - AI Prediction System</h1>
<h3>BOSS's AI System - Day 5 Final Project</h3>
<hr>
<h2>Sales Prediction</h2>
<form method="post">
    Quantity (10-100): <input type="number" name="qty" value="50">
    <input type="submit" name="btn1" value="Predict Sale">
</form>
<h3>{{ sale_result }}</h3>
<hr>
<h2>Customer Buy Prediction</h2>
<form method="post">
    Age (18-70): <input type="number" name="age" value="28">
    <input type="submit" name="btn2" value="Will Buy?">
</form>
<h3>{{ buy_result }}</h3>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    sale_result = ""
    buy_result = ""
    if request.method == "POST":
        if "btn1" in request.form:
            qty = int(request.form["qty"])
            pred = model1.predict([[qty]])
            sale_result = f"Boss, {qty} qty pe Sale hogi: Rs. {int(pred[0]):,}"
        if "btn2" in request.form:
            age = int(request.form["age"])
            res = model2.predict([[age]])[0]
            if res == 1:
                buy_result = f"{age} years customer WILL BUY!"
            else:
                buy_result = f"{age} years customer WON'T BUY!"
    return render_template_string(HTML, sale_result=sale_result, buy_result=buy_result)

if __name__ == "__main__":
    app.run(debug=True)