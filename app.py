from flask import Flask,request,render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__,template_folder='template')
@app.route('/', methods=['GET', 'POST'])
def main():
    
    if request.method == "POST":

        #load the data and fit the model.
        data = pd.read_excel('Historical Alarm Cases.xlsx')
        X = data.iloc[:, 1:7]
        Y = data['Spuriosity Index(0/1)']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
        lm = LogisticRegression()
        model = lm.fit(X_train, Y_train)

        #get data from user.    
        f1 = int(request.form.get('Ambient Temperature'))
        f2 = float(request.form.get('Calibration'))
        f3 = int(request.form.get('Unwanted substance deposition'))
        f4 = int(request.form.get('Humidity'))
        f5 = int(request.form.get('H2S Content'))
        f6 = int(request.form.get('detected by'))
        data = [f1, f2, f3, f4, f5, f6]
        test_data = np.array(data).reshape(1,6)
        
        #get prediction.
        prediction = model.predict(test_data)
        
        if prediction == 1:
            result = "True Alarm, It's Danger...!"
            return render_template("file.html", output = result)
        else:
            result = "False Alarm, Relax...!"
            return render_template("file.html", output = result)
        

    else:
        prediction = ""

    return render_template("file.html", output = prediction)

if __name__ == '__main__':
    app.run()

