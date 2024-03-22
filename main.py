# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
#from plotly import graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import urllib.request
import urllib.parse
import csv
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="money_laundering"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""


    return render_template('index.html',msg=msg)


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)



@app.route('/login_cus', methods=['GET', 'POST'])
def login_cus():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM reg_mgr WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_cus.html',msg=msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
        
    mycursor = mydb.cursor()
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        
        mycursor.execute("SELECT count(*) FROM user_reg where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]
        
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM user_reg")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO user_reg(id,name,mobile,email,uname,pass,rdate) VALUES (%s, %s, %s, %s, %s, %s,%s)"
            val = (maxid,name,mobile,email,uname,pass1,rdate)
            mycursor.execute(sql, val)
            mydb.commit()
            
            msg="success"
            
        else:
            msg='fail'
    return render_template('register.html',msg=msg)

@app.route('/reg_mgr', methods=['GET', 'POST'])
def reg_mgr():
    #import student
    msg=""
    mess=""
    email=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        branch=request.form['branch']
        uname=request.form['uname']
        pass1=request.form['pass']

        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM reg_mgr where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]
        print("ff")
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM reg_mgr")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO reg_mgr(id,name,mobile,email,branch,uname,pass) VALUES (%s, %s, %s, %s, %s, %s,%s)"
            val = (maxid,name,mobile,email,branch,uname,pass1)
            mycursor.execute(sql, val)
            mydb.commit()
            mess="RMI Agent Username:"+uname+", Password:"+pass1
            #print(mycursor.rowcount, "Registered Success")
            msg="success"
            
        else:
            msg='fail'
    return render_template('reg_mgr.html',msg=msg,mess=mess,email=email)

@app.route('/view_mgr', methods=['GET', 'POST'])
def view_mgr():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM reg_mgr")
    data = mycursor.fetchall()


    
    return render_template('view_mgr.html',msg=msg,data=data)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM alert_info order by id desc")
    data = mycursor.fetchall()


    
    return render_template('detect.html',msg=msg,data=data)

@app.route('/view_cus', methods=['GET', 'POST'])
def view_cus():
    msg=""
    act=request.args.get("act")
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM reg_mgr where uname=%s",(uname,))
    usr = mycursor.fetchone()
    
    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from user_reg where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('view_cus'))
        
    mycursor.execute("SELECT * FROM user_reg")
    data = mycursor.fetchall()


    
    return render_template('view_cus.html',msg=msg,data=data,usr=usr)


@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    cnt=0
    
    st=""
    result=""
    uname=""
    act = request.args.get('act')
    cat = request.args.get('cat')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM reg_mgr where uname=%s",(uname,))
    usr = mycursor.fetchone()



    
    return render_template('userhome.html',msg=msg,usr=usr)

@app.route('/test_data', methods=['GET', 'POST'])
def test_data():
    msg=""
    uname=""
    act = request.args.get('act')

    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM reg_mgr where uname=%s",(uname,))
    usr = mycursor.fetchone()

    
    return render_template('test_data.html',msg=msg,usr=usr)

@app.route('/page', methods=['GET', 'POST'])
def page():
    msg=""
    uname=""
    data1=[]
    data2=[]
    act = request.args.get('act')

    df = pd.read_csv("static/dataset/Fraud1.csv")
    data2=[]

    ff=open("detect.txt","w")
    ff.write("no")
    ff.close()

    rn=randint(5,3700)
    print(rn)
    i=0
    k=rn
    n=rn+10
    for dd in df.values:
        if i==k:
            if k<=n:
                
                if dd[1]=="TRANSFER" or dd[1]=="PAYMENT":
                    dt=[]
                    dt.append(dd[0])
                    dt.append(dd[1])
                    dt.append(dd[2])
                    dt.append(dd[3])
                    dt.append(dd[5])

                    a1=float(dd[5])
                    a2=float(dd[2])
                    bal=a1+a2

                        
                    dt.append(bal)
                    dt.append(dd[6])
                    dt.append(dd[7])
                    dt.append(dd[8])
                    dt.append(dd[9])
                    
                    data1.append(dt)
                    if dd[9]==1:
                        a1=float(dd[5])
                        a2=float(dd[2])
                        bal=a1+a2
                        
                        vv="Credit-"+str(dd[2])+"-"+dd[3]+"-"+dd[6]+"-"+str(dd[5])+"-"+str(bal)

                        ff=open("detect.txt","w")
                        ff.write("credit")
                        ff.close()

                        ff=open("value.txt","w")
                        ff.write(vv)
                        ff.close()
                k+=1

        i+=1
    
        
    return render_template('page.html',msg=msg,data1=data1,data2=data2)


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    msg=""
    uname=""
    data1=[]
    data2=[]
    act = request.args.get('act')

    df = pd.read_csv("static/dataset/Fraud1.csv")
    data2=[]

    ff=open("detect.txt","w")
    ff.write("no")
    ff.close()

    #3230
    rn2=3230
    #randint(5,3700)
    j=0
    h=rn2
    n2=rn2+10
    for dd2 in df.values:
        if j==h:
            if h<=n2:
                if dd2[1]=="CASH_OUT" or dd2[1]=="DEBIT":
                    dt=[]
                    dt.append(dd2[0])
                    dt.append(dd2[1])
                    dt.append(dd2[2])
                    dt.append(dd2[3])
                    dt.append(dd2[4])                    
                    dt.append(dd2[5])
                    dt.append(dd2[6])
                    dt.append(dd2[8])

                    a1=float(dd2[8])
                    a2=float(dd2[2])
                    bal=a1-a2

                        
                    dt.append(bal)
                    
                    dt.append(dd2[9])
                    
                    data2.append(dt)
                    
                    if dd2[9]==1:

                        a1=float(dd2[8])
                        a2=float(dd2[2])
                        bal=a1-a2
                        
                        
                        vv="Debit-"+str(dd2[2])+"-"+dd2[3]+"-"+dd2[6]+"-"+str(dd2[8])+"-"+str(bal)

                        ff=open("detect.txt","w")
                        ff.write("debit")
                        ff.close()

                        ff=open("value.txt","w")
                        ff.write(vv)
                        ff.close()
                    
                h+=1
        j+=1


        
    return render_template('page2.html',msg=msg,data1=data1,data2=data2)


@app.route('/page3', methods=['GET', 'POST'])
def page3():
    msg=""
    uname=""
    mess=""
    email=""
    st=""
    data1=[]
    data2=[]
    mycursor = mydb.cursor()
    act = request.args.get('act')

    ff=open("email.txt","r")
    email=ff.read()
    ff.close()
    

    ff=open("detect.txt","r")
    res=ff.read()
    ff.close()

    ff=open("value.txt","r")
    value=ff.read()
    ff.close()

    if res=="credit":
        st="1"
        v1=value.split("-")
         
        mess="Amount Rs. "+v1[1]+", credited from "+v1[2]+" to "+v1[3]+", Previous balance: "+v1[4]+", after credit:"+v1[5]
        ff=open("detect.txt","w")
        ff.write("no")
        ff.close()
    elif res=="debit":
        st="1"
        v1=value.split("-")         
        mess="Amount Rs. "+v1[1]+", debited from "+v1[2]+" to "+v1[3]+", Previous balance: "+v1[4]+", after debit:"+v1[5]
        ff=open("detect.txt","w")
        ff.write("no")
        ff.close()

    if mess=="":
        s=1
    else:
        mycursor.execute("SELECT max(id)+1 FROM alert_info")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO alert_info(id,message) VALUES (%s, %s)"
        val = (maxid,mess)
        mycursor.execute(sql, val)
        mydb.commit()
        
        msg="success"

        
        
    return render_template('page3.html',msg=msg,st=st,mess=mess,email=email,res=res)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    ff=open("email.txt","r")
    email=ff.read()
    ff.close()
    
    if request.method=='POST':
        email=request.form['email']
        ff=open("email.txt","w")
        ff.write(email)
        ff.close()
        msg="ok"
    
    return render_template('admin.html',msg=msg,email=email)

@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    msg=""

    application = pd.read_csv("static/dataset/Fraud1.csv")
    dat=application.head(50)
    data2=[]
    for dd in dat.values:
        data2.append(dd)
    
    
    return render_template('view_data.html',data2=data2)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    msg=""
    mem=0
    cnt=0
    cols=0
    filename = 'static/dataset/Fraud1.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())
    cname=[]
    data=[]
    dtype=[]
    dtt=[]
    nv=[]
    i=0
    
    sd=len(data1)
    rows=len(data1.values)
    
    #print(data1.columns)
    col=data1.columns
    #print(data1[0])
    for ss in data1.values:
        cnt=len(ss)
        

    i=0
    while i<cnt:
        j=0
        x=0
        for rr in data1.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt.append(dt)
        nv.append(str(x))
        
        i+=1

    arr1=np.array(col)
    arr2=np.array(nv)
    data3=np.vstack((arr1, arr2))


    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data=np.vstack((arr3, arr4))
   
    print(data)
    cols=cnt
    mem=float(rows)*0.75

    return render_template('preprocess.html',data=data, msg=msg, rows=rows, cols=cols, dtype=dtype, mem=mem)

    

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""
    data2=[]

    filename = 'static/dataset/Fraud1.csv'
    df = pd.read_csv(filename, header=0)
    dd=df.describe()

    from sklearn.model_selection import train_test_split

    df_sample, _ = train_test_split(df, test_size=0.92141601,random_state=1234, stratify=df["isFraud"])

    df_train, df_test = train_test_split(df_sample, test_size=0.2,random_state=1234,stratify=df_sample["isFraud"])
    df_train, df_val = train_test_split(df_train, test_size=0.25,random_state=1234,stratify=df_train["isFraud"])
    df_train.isnull().sum()
    df_train.shape

    from sklearn.preprocessing import StandardScaler, RobustScaler

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df_train['scaled_amount'] = rob_scaler.fit_transform(df_train['amount'].values.reshape(-1,1))
    df_train['scaled_time'] = rob_scaler.fit_transform(df_train['step'].values.reshape(-1,1))

    df_train['isFraud'].value_counts()
    '''sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="isFraud", data=df_train , color = "midnightblue")
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))

    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    df_train.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8) , color = "midnightblue")
    plt.ticklabel_format(style='plain', axis='y')'''

    dd=pd.crosstab(df_train['type'], df_train['isFraud'])
    #print(dd)
    

    '''ax=sns.countplot('type', data=df_train[(df_train['isFraud'] == 1)])
    plt.title('Fraud Distribution', fontsize=14)
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))

    #plt.show()

    df_train['transactionHour'] = df_train['step'] % 24

    frauds_hour = pd.concat([df_train.groupby("transactionHour")["isFraud"].sum(),df_train.groupby("transactionHour")["isFraud"].count()],axis=1)
    frauds_hour.columns = ["Frauds","Transactions"]
    frauds_hour["fraud_rate"] = frauds_hour.Frauds/frauds_hour.Transactions
    sns.barplot(x=frauds_hour.index,y=frauds_hour.fraud_rate)
    #plt.show()

    # converting into object type
    df_train['transactionHour'] = df_train['transactionHour'].astype('object')
    ##
    plt.figure(figsize=(18,6))
    sns.lineplot(data=df_train.groupby(['transactionHour','type']).agg({'amount' : 'mean'}).round(2).reset_index(),
                 x='transactionHour',
                 y='amount',
                 hue='type')
    plt.xlabel('Transaction hour', fontsize=15, fontweight='bold')
    plt.xticks(range(24), range(24),fontsize=15, fontweight='bold', rotation=0)
    plt.ylabel('Average transaction amount (millions)', fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.title('Average transaction amount at each hour', fontsize=22, fontweight='bold')'''
    


    
    return render_template('process1.html',data2=data2,dd=dd)

@app.route('/process2', methods=['GET', 'POST'])
def process2():

    '''filename = 'static/dataset/Fraud1.csv'
    df = pd.read_csv(filename, header=0)
    dd=df.describe()

    from sklearn.model_selection import train_test_split

    df_sample, _ = train_test_split(df, test_size=0.92141601,random_state=1234, stratify=df["isFraud"])

    df_train, df_test = train_test_split(df_sample, test_size=0.2,random_state=1234,stratify=df_sample["isFraud"])
    df_train, df_val = train_test_split(df_train, test_size=0.25,random_state=1234,stratify=df_train["isFraud"])
    df_train.isnull().sum()
    df_train.shape

    from sklearn.preprocessing import StandardScaler, RobustScaler

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df_train['scaled_amount'] = rob_scaler.fit_transform(df_train['amount'].values.reshape(-1,1))
    df_train['scaled_time'] = rob_scaler.fit_transform(df_train['step'].values.reshape(-1,1))

    df_train['isFraud'].value_counts()
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="isFraud", data=df_train , color = "midnightblue")
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))

    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    df_train.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8) , color = "midnightblue")
    plt.ticklabel_format(style='plain', axis='y')

    dd=pd.crosstab(df_train['type'], df_train['isFraud'])
    print(dd)
    

    ax=sns.countplot('type', data=df_train[(df_train['isFraud'] == 1)])
    plt.title('Fraud Distribution', fontsize=14)
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))

    #plt.show()

    df_train['transactionHour'] = df_train['step'] % 24

    frauds_hour = pd.concat([df_train.groupby("transactionHour")["isFraud"].sum(),df_train.groupby("transactionHour")["isFraud"].count()],axis=1)
    frauds_hour.columns = ["Frauds","Transactions"]
    frauds_hour["fraud_rate"] = frauds_hour.Frauds/frauds_hour.Transactions
    sns.barplot(x=frauds_hour.index,y=frauds_hour.fraud_rate)
    #plt.show()

    # converting into object type
    df_train['transactionHour'] = df_train['transactionHour'].astype('object')
    ##
    plt.figure(figsize=(18,6))
    sns.lineplot(data=df_train.groupby(['transactionHour','type']).agg({'amount' : 'mean'}).round(2).reset_index(),
                 x='transactionHour',
                 y='amount',
                 hue='type')
    plt.xlabel('Transaction hour', fontsize=15, fontweight='bold')
    plt.xticks(range(24), range(24),fontsize=15, fontweight='bold', rotation=0)
    plt.ylabel('Average transaction amount (millions)', fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.title('Average transaction amount at each hour', fontsize=22, fontweight='bold')
    ##################
    newscatplot=df_train[df_train['isFraud']==1]
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_ylim(0,2*1e7)
    ax.set_xlim(0,2*1e7)
    df_train.plot.scatter(x='oldbalanceOrg',y='amount', ax=ax,edgecolors='red',s=100,alpha=0.1,label="Legit transaction")
    newscatplot.plot.scatter(x='oldbalanceOrg',y='amount', color='#FCD735', ax=ax,edgecolors='red',s=100,alpha=0.1,label="Fraud transcation")
    plt.title('Amount vs Balance',fontsize=25,color='#E43A36')

    df_train = pd.concat([df_train, pd.get_dummies(df_train['type'], prefix='type')],axis=1)
    df_train = df_train.drop(['type'],axis=1)
    df_train = df_train.drop(["isFlaggedFraud"],axis=1)
    df_train = df_train.drop(['nameOrig',"nameDest"],axis=1)
    y_train = df_train["isFraud"]
    X_train = df_train.drop(['isFraud'],axis=1)

    df['Class'].value_counts()
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="Class", data=df,color = "midnightblue" )
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')),
                (p.get_x(), p.get_height()))

    ##
    df_train, df_test = train_test_split(df, test_size=0.2,random_state=123,stratify=df["Class"])
    df_train, df_val = train_test_split(df_train, test_size=0.25,random_state=123,stratify=df_train["Class"])


    plt.figure(figsize = (8,6))
    plt.title('Credit Card Transactions features correlation plot (Pearson)')
    corr = df_train.corr()
    sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="PuBuGn_r")
    plt.show()'''

    data=pd.read_csv('static/dataset/Fraud1.csv')
    data1=data.describe().T

    #sns.heatmap(data.corr(),annot=True,cmap="PiYG")

   
    return render_template('process2.html',data1=data1)

##LSTM
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

##########################
    
@app.route('/process3', methods=['GET', 'POST'])
def process3():
    application = pd.read_csv("static/dataset/Fraud.csv")
    dat=application.head(50)
    data2=[]
    for dd in dat.values:
        data2.append(dd)
    
    data=pd.read_csv('static/dataset/Fraud1.csv')
    data1=data.describe().T
    df1=data.copy()
    df1['type'].unique()
    df1['type']=df1['type'].map({'PAYMENT':1 ,'TRANSFER':2, 'DEBIT':3, 'CASH_IN':4, 'CASH_OUT':5})
    df2=df1.drop(['nameOrig','nameDest'],axis=1)
    df3=df2.drop('newbalanceOrig',axis=1)
    df4=df3.drop('newbalanceDest',axis=1)
    print(data.groupby('type')['isFraud'].sum())

    fraud_trans=data.loc[(data['isFraud']==1) & (data['type']=='TRANSFER')]
    fraud_cashout=data.loc[(data['isFraud']==1) & (data['type']=='CASH_OUT')]
    print('number of fraudulent transactions with type transfer',len(fraud_trans))
    print('number of fraudulent transactions with type cashout',len(fraud_cashout))
    data["isFraud"].value_counts().plot(kind='pie',autopct='%.3f%%')
    

    

    return render_template('process3.html',data2=data2)



##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


