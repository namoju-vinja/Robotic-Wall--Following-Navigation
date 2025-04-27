from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile

# Create your views here.

def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')

#####machine learning
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

global X_train,X_test,y_train,y_test
X_train = None
scaler = StandardScaler()
def Upload_data(request):
    load=True
    global X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path))
        le = LabelEncoder()
        df['Class'] = le.fit_transform(df['Class'])
        X= df.iloc[:, :-1]
        y= df.iloc[:, -1]
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
        X_train = scaler.fit_transform(X_train)

        # Transform the testing data using the same scaler
        X_test = scaler.transform(X_test)
        default_storage.delete(file_path)
        print('---done---')
        outdata=df.head(100)
        return render(request,'prediction.html',{'predict':outdata.to_html()})
    return render(request,'prediction.html',{'upload':load})


#defining global variables to store accuracy and other metrics

labels=['Slight-Right-Turn', 'Sharp-Right-Turn', 'Move-Forward','Slight-Left-Turn']
precision = []
recall = []
fscore = []
accuracy = []#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def Knn(request):
    modelfile2 = 'model/KNN_classifier_weights.pkl'
    if os.path.exists(modelfile2):
        # Load the model from the pkl file
        classifier= joblib.load(modelfile2)
        predict = classifier.predict(X_test)
        print("KNN Model trained weights loaded.")
        calculateMetrics("KNN Classifier", predict, y_test)
    else:
        classifier = KNeighborsClassifier()
        # Train the classifier on the training data
        classifier.fit(X_train, y_train)
        # Make predictions on the test data
        predict=classifier.predict(X_test)
        joblib.dump(classifier, modelfile2)
        print("KNN Model trained and model weights saved.")
        calculateMetrics("KNN Tree Classifier", predict, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'KNN Classifier',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
def dtc(request):
    modelfile2 = 'model/Dt_classifier_weights.pkl'
    if os.path.exists(modelfile2):
        # Load the model from the pkl file
        dt_classifier= joblib.load(modelfile2)
        predict = dt_classifier.predict(X_test)
        calculateMetrics("DecisionTreeClassifier", predict, y_test)
    else:
        dt_classifier = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=15)
        # Train the classifier on the training data
        dt_classifier.fit(X_train, y_train)
        # Make predictions on the test data
        predict=dt_classifier.predict(X_test)
        joblib.dump(dt_classifier, modelfile2)
        print("dt_classifier_model trained and model weights saved.")
        calculateMetrics("Decision Tree Classifier", predict, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'Decision Tree Classifier',
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})

def prediction_view(request):
    import joblib
    Test=True
    modelfile2 = 'model/Dt_classifier_weights.pkl'
    dt_classifier= joblib.load(modelfile2)
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        testdata = pd.read_csv(default_storage.path(file_path))
        test = testdata
        default_storage.delete(file_path)
        scaled_test=scaler.fit_transform(test)
        predict = dt_classifier.predict(scaled_test)
        # Loop through each prediction and print the corresponding row
        data = []  # This will hold the rows and results
        for i, p in enumerate(predict):
            row_data = test.iloc[i]  # Get the row data
            data.append({
                'row':row_data,
                'message':f"Row {i}:************************************************** {labels[p]}"
                    })  # Print the row
        return render(request, 'prediction.html', {'data': data})
    return render(request,'prediction.html',{'test':Test})