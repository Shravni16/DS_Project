import streamlit as st
import pandas as pd 
import numpy as nd
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['agg.path.chunksize'] = 1000000000000000000

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


rad =st.sidebar.radio("Navigation",["DashBoard","Detector"])

df = pd.read_csv("optimized_dataset.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)
def Corealtion():
    st.header("Corelation Table :")
    st.table(df.corr(numeric_only=True))

def HeatMap():
    st.header("HeatMap :")
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr() , annot = True)
    st.pyplot()

def DataBeforeNormalization():
    st.header("Data Before Normalization :")
    X_train_b_n.plot()
    st.pyplot()

def DataAfterNormalization():
    st.header("Data After Normalization :")
    X_train.plot()
    plt.xlim(0,200)
    st.pyplot()

def TrainingAccuracy():
    st.header("Training Accuracy :")
    st.header(accuracy_score(y_train, y_out_knn))

def TestingAccuracy():
    st.header("Testing Accuracy :")
    st.header(accuracy_score(y_test, y_pred_knn))


def ConfusionMatrix():
    st.header("Confusion Matrix :")
    log_CM = confusion_matrix(y_test, y_pred_knn, labels = [0, 1])
    plt.figure(figsize=(7,5))
    sns.heatmap(log_CM, annot = True)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.title(" Confusion Matrix", fontsize=15,c="Green")
    st.pyplot()

def ClassificationReport():
    from sklearn.metrics import classification_report
    st.header("Classification Report :")
    CR=classification_report(y_test, y_pred_knn)
    st.table(CR)

y=df['label']
X=df.drop(['label'] ,axis='columns')

# Train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_b_n=X_train

# Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_cols=X_train.columns

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train=pd.DataFrame(X_train)

X_train.columns=X_train_cols

#Fitting data in KNN

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=13,weights='uniform',metric= 'manhattan')

knn_model.fit(X_train, y_train)

y_out_knn = knn_model.predict(X_train)

y_pred_knn = knn_model.predict(X_test)

if rad=="DashBoard":
    choice = st.radio("Visualizations",["Corelation","HeatMap","Data before Normalization","Data after Normalization","Training Accuracy","Testing Accuracy","Confusion matrix","Classification Report"],index = 2)
    if choice=="Corelation":
        Corealtion()
    elif choice=="HeatMap":
        HeatMap()
    elif choice=="Data before Normalization":
        DataBeforeNormalization()
    elif choice=="Data after Normalization":
        DataAfterNormalization()
    elif choice=="Training Accuracy":
        TrainingAccuracy()
    elif choice=="Testing Accuracy":
        TestingAccuracy()
    elif choice=="Confusion matrix":
        ConfusionMatrix()
    elif choice=="Classification Report":
        ClassificationReport()

if rad=="Detector":
    st.title("Bank Fraud Detector")
    col1,col2,col3=st.columns(3)
    last_rech=col1.text_input("Last Recharge Amount")
    median_amnt_loan=col2.text_input("Median Amount Loan")
    Daily_dcr=col3.text_input("Daily Decrement")

    col4,col5,col6=st.columns(3)
    last_rech_date=col4.text_input("Last Date of Recharge")
    rech_count=col5.text_input("Recharge Count")
    cnt_loan=col6.text_input("Loans Count")

    col7,col8,col9=st.columns(3)
    total_amt_loan=col7.text_input("Total Loan Amount")
    max_loan=col8.text_input("Maximum Loan Amount")
    payback_r=col9.text_input("Payback Ratio")

    if st.button("Submit"):
        X_test_inp_temp=[last_rech,median_amnt_loan,Daily_dcr,last_rech_date,rech_count,cnt_loan,total_amt_loan,max_loan,payback_r]
        X_test_inp=nd.array(X_test_inp_temp)
        X_test_inp=X_test_inp.reshape(1,9)
        X_test_inp=pd.DataFrame(X_test_inp)
        X_test_inp = scaler.transform(X_test_inp)

        y_pred_inp=knn_model.predict(X_test_inp)

        if y_pred_inp==1:
            st.warning("Transaction is Fraud!")
            st.markdown("""
                        <style>
                            .warning {
                                color: #FFD700; /* Yellow color code */
                            }
                        </style>
                        """, unsafe_allow_html=True)
        else:
            st.success("Transaction is not Fraud.")
            st.markdown("""
                    <style>
                        .green-checkbox {
                            color: #008000; /* Green color code */
                        font-size: 30px; /* Adjust font size as needed */
                        }
                    </style>
                    """, unsafe_allow_html=True)
        st.success("Succesfully Executed")



