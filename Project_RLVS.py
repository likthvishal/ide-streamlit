import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve,plot_confusion_matrix
from sklearn.metrics import confusion_matrix




from sklearn.metrics import precision_score, recall_score 

import matplotlib.pyplot as plt
import seaborn as sb



def main():

	st.title("SURVIVING ON TITANIC ")
	st.markdown("Did you ever imagine yourself being on titanic?")
	st.sidebar.title("CHOOSE WISE")
	st.sidebar.markdown("make better solutions")

	@st.cache(persist=True)
	def load_data():
		data = pd.read_csv("titanic.csv")
		cols = ['Name', 'Ticket', 'Cabin']
		data = data.drop(cols, axis=1)
		data = data.dropna()
		dummies = []
		cols = ['Pclass', 'Sex', 'Embarked']
		for col in cols:
			dummies.append(pd.get_dummies(data[col]))

		titanic_dummies = pd.concat(dummies, axis=1)
		data = pd.concat((data,titanic_dummies), axis=1)
		data = data.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
		data['Age'] = data['Age'].interpolate()




		return data

	@st.cache(persist=True)
	def split(df):
		x = df.drop(columns =['Survived'])
		y = df.Survived
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
		return x_train, x_test, y_train, y_test


	def plot_metric(metrics_list):
		if 'Confusion_matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC_curve' in metrics_list:
			st.subheader("ROC_curve")
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()	

		if 'Precession-recall' in metrics_list:
			st.subheader("Precession-recall-Curve")
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()


	df = load_data()
	x_train, x_test, y_train, y_test = split(df)

	class_names = ['Survived','Not Survived']




	if st.sidebar.checkbox("show raw data", False):
		st.subheader("Titanic data (source Kaggle)")
		st.write(df)



	st.sidebar.subheader("choose classifier")

	classifier = st.sidebar.selectbox("classifier_algo", ("None","SVM","LogisticRegression","RandomForestClassifier"))

	



	if classifier =="SVM":
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (regularization)",0.01,10.0,step=0.01, key="C")
		kernel = st.sidebar.radio("kernel",("rbf","linear"),key = 'kernel')
		gamma  = st.sidebar.radio("gamma (kernel coefficient)",("scale","auto"), key= 'gamma')

		metrics = st.sidebar.multiselect("what metric to plot?",("Confusion_matrix","ROC_curve","Precession-recall"))

		if st.sidebar.button("Classify", key='classify'):

			st.subheader( "Support Vector Machine (SVM) Results") 
			model = SVC(C=C, kernel=kernel, gamma=gamma) 
			model.fit (x_train, y_train) 
			accuracy = model.score(x_test, y_test) 
			y_pred = model.predict(x_test) 
			st.write("Accuracy:",accuracy.round(2))
			cm_dtc=confusion_matrix(y_test,y_pred)
			st.write("Confusion_matrix:",cm_dtc)
			st.write("precision: ",precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
			plot_metric(metrics)


	if classifier == "LogisticRegression":
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (regularization parameter)",0.01,10.0, step = 0.01, key='C_LR')
		max_iter = st.sidebar.slider("Maximum number of iterations",100,500, key='max_iter')



		metrics = st.sidebar.multiselect("what metric to plot?",("Confusion_matrix","ROC_curve","Precession-recall"))

		if st.sidebar.button("Classify", key='classify'):

			st.subheader( "Logistic Regression Results") 
			model =LogisticRegression(C=C,max_iter=max_iter) 
			model.fit (x_train, y_train) 
			accuracy = model.score(x_test, y_test) 
			y_pred = model.predict(x_test) 
			st.write("Accuracy:",accuracy.round(2))
			cm_dtc=confusion_matrix(y_test,y_pred)
			st.write("Confusion_matrix:",cm_dtc)
			st.write("precision: ",precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
			plot_metric(metrics)



	if classifier == "RandomForestClassifier":
		st.sidebar.subheader("Model Hyperparameters")
		n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000, step = 10, key='n_estimators')
		max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20, step = 1, key="max_depth")
		bootstrap = st.sidebar.radio("bootstrap samples when building trees", ('True','False'), key='bootstrap')


		metrics = st.sidebar.multiselect("what metric to plot?",("Confusion_matrix","ROC_curve","Precession-recall"))

		if st.sidebar.button("Classify", key='classify'):

			st.subheader( "Random Forest Classifier Results")
			model =RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1) 
			model.fit (x_train, y_train) 
			accuracy = model.score(x_test, y_test) 
			y_pred = model.predict(x_test) 
			st.write("Accuracy:",accuracy.round(2))
			cm_dtc=confusion_matrix(y_test,y_pred)
			st.write("Confusion_matrix:",cm_dtc)
			st.write("precision: ",precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
			plot_metric(metrics)




	if st.sidebar.checkbox("Compare model performance"):
		st.sidebar.title("SVM")
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (regularization)",0.01,10.0,step=0.01, key="C")
		kernel = st.sidebar.radio("kernel",("rbf","linear"),key = 'kernel')
		gamma  = st.sidebar.radio("gamma (kernel coefficient)",("scale","auto"), key= 'gamma')
		model = SVC(C=C, kernel=kernel, gamma=gamma) 
		model.fit (x_train, y_train) 
		ac_svm = model.score(x_test, y_test)

		st.sidebar.title("Logistic Regression")
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (regularization parameter)",0.01,10.0, step = 0.01, key='C_LR')
		max_iter = st.sidebar.slider("Maximum number of iterations",100,500, key='max_iter')
		model =LogisticRegression(C=C,max_iter=max_iter) 
		model.fit (x_train, y_train) 
		ac_lr = model.score(x_test, y_test)


		st.sidebar.title("Random Forest Classifier")
		n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000, step = 10, key='n_estimators')
		max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20, step = 1, key="max_depth")
		bootstrap = st.sidebar.radio("bootstrap samples when building trees", ('True','False'), key='bootstrap')
		model =RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1) 
		model.fit (x_train, y_train) 
		ac_rfc = model.score(x_test, y_test)

		values_ = [ac_svm,ac_rfc,ac_lr]
		val = ['SVM','RandomForestClassifier','LogisticRegression']


		fig, ax = plt.subplots()
		ax.bar(val,values_)
		plt.title("Models comparison", fontsize=16)
		plt.xlabel("Model name")
		plt.ylabel("Accuracy")
		st.pyplot(fig)



	if st.sidebar.checkbox("Data Visualization",False):

		plotting = st.sidebar.selectbox("data Visualization", ("None","Fareprice","Age_and_Sex","BarPlots(fare price)","Histogram(Fare)","scatter","survived"))


		if plotting == "Age_and_Sex":
			fig,ax = plt.subplots()
			ax = sb.countplot('Sex',hue='Survived',data=df,)

			st.pyplot(fig)

		


		if plotting == "scatter":
			fig, ax = plt.subplots()
			ax.scatter(df['Age'],df['Fare'],alpha=0.5)
			ax.grid(True)
			fig.tight_layout()
			plt.title("Fare & Age",fontsize=16)

			st.pyplot(fig)

		if plotting == "Fareprice":
			fig,ax = plt.subplots()
			ax.plot(df['Fare'])
			plt.title("Price distribution",fontsize=20)

			st.pyplot(fig)




		if plotting == "survived":
			ax,fig = plt.subplots()
			fig = sb.countplot('Survived',data=df)
			plt.title("Survival", fontsize=16)

			st.pyplot(ax)





			





		if plotting == "Histogram(Fare)":
		
			fig, ax = plt.subplots()
			ax.hist(df["Fare"], bins=20)
			plt.xticks(fontsize=14)
			plt.yticks(fontsize=14)
			plt.title("Fare distribution", fontsize=16)

			st.pyplot(fig)



		if plotting == "BarPlots(fare price)":



			fig, ax = plt.subplots()
			embarked_info = df["Embarked"].value_counts()
			ax.bar(embarked_info.index, embarked_info.values)
			plt.xticks(fontsize=14)
			plt.yticks(fontsize=14)
			plt.title("Embarked distribution", fontsize=16)

			st.pyplot(fig)







		





	















if __name__  == '__main__':
	main()