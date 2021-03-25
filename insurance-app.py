import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from PIL import Image 
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

plt.style.use('seaborn-notebook')
sns.set(context="paper", font="monospace")


# this result would help insurance companies and public health 
# stakeholders make smarter decisions on targeted ads to potential customers and improve health 
# insurance penetration amongst the Nigerian populace. 

def main():
    page = st.sidebar.selectbox("Choose a page",["About App","Insurance Subscription Analysis","Machine Learning"])

    if page =="About App":
        st.title('Analysis and Prediction of Health Insurance Subscription in Nigeria')
        image = Image.open('images/GB.png')
        st.image(image)
        # st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6MWvMt4EgDEKS4S6Y4z_HGlIYmUomW9Llg&usqp=CAU.png", 
        # 	width=480)
        
        st.markdown("""
		This app predicts whether an individual would take up an health insurance policy or not leveraging 
		a machine learning model. We would also investigate factors that most likely influence taking up an health 
		policy using the trained model. 

		Data obtained from Individual Recode section of the 2018 Nigerian Demographic and
		 Health Survey [DHS](https://dhsprogram.com/data/dataset/Nigeria_Standard-DHS_2018.cfm) .
		""")

        st.markdown("## Meet the Data Scientists")

        col1,mid,col2 = st.beta_columns(3)
        with col1:
            st.image('images/ironman2.jpg',width=300)
            html = f"Opeyemi Idris <a href='https://github.com/hardcore05' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.image('images/ironman2.jpg',width=300)
            html = f"Shakiru Muraina <a href='https://github.com/Debare' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        col1,mid,col2 = st.beta_columns(3)
        with col1:
            st.image('images/ironman2.jpg',width=300)
            html = f"Boluwatife Adewale <a href='https://github.com/BBLinus' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.image('images/ironman2.jpg',width=300)
            html = f"Uthman Allison <a href='https://github.com/alliwene' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)

    if page == "Insurance Subscription Analysis":
        st.title("Explore Your Dataset")
        # data = st.file_uploader("Only csv files allowed",type=['csv'])

        # if data:
        #     data = pd.read_csv(data)

        #     st.title("Look at the DataFrame")
        #     st.dataframe(data)

        #     with st.beta_expander("Visualize the data?"):
        #         dim=(15.0,10.0)
        #         fig = plt.figure(figsize=dim)

        #         viz_page = st.sidebar.selectbox('choose visual',['Count Plot','Bar Chart', 'Pie chart'])

        #         if viz_page == "Count Plot":
        #             columns = data.columns
        #             x = st.selectbox('choose column',columns)
                    
                    
        #             sns.countplot(x=x,data=data)


        #             st.pyplot(fig)


    if page == "Machine Learning":
        st.title("Machine Learning")
        st.sidebar.header('User Input Features')

        st.sidebar.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/alliwene/gb-november-grp2-health-insurance/main/data/data_sample.csv?token=AHCUSLFXQUYKYB4LQYLY2PTAMYJUM)
        """)

		# Collects user input features into dataframe
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)

        # else:
        #     def user_input_features():
        #         data = {}     
        #         features = pd.DataFrame(data, index=[0])
        #         return features
        #     input_df = user_input_features()
		    #     island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
		    #     sex = st.sidebar.selectbox('Sex',('male','female'))
		    #     bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
		    #     bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
		    #     flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
		    #     body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
		    #     data = {'island': island,
		    #             'bill_length_mm': bill_length_mm,
		    #             'bill_depth_mm': bill_depth_mm,
		    #             'flipper_length_mm': flipper_length_mm,
		    #             'body_mass_g': body_mass_g,
		    #             'sex': sex}
                

        # Combines user input features with cleaned dataset
        # This will be useful for the encoding phase
        # insurance_clean = pd.read_csv('data_clean.csv')
        # insurance = insurance_clean.drop(columns=['target'])
        # df = pd.concat([input_df,insurance],axis=0)

            # get categorical features of input_df
            cat_feat = input_df.select_dtypes(exclude = np.number).columns 

            # label encode categorical features 
            for feat in cat_feat:
                input_df[feat] = pd.factorize(input_df[feat])[0]
            # df = df[:1] # Selects only the first row (the user input data)
            # st.write(input_df)

            # Displays the user input features
            st.subheader('User Input features')
            if uploaded_file is not None:
                st.write(input_df)
        # else:
        #     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        #     st.write(input_df)
            # Reads in saved classification model
            load_clf = pickle.load(open('model/insurance_rf.pkl', 'rb'))

            # Apply model to make predictions
            prediction = load_clf.predict(input_df)
            prediction_proba = load_clf.predict_proba(input_df)

            st.subheader('Prediction')
            output = np.array(['Yes','No'])
            st.write(output[prediction])

            st.subheader('Prediction Probability')
            st.write(prediction_proba)

            # make feature importance plot 
            st.subheader('Feature Importance Plot')
            feat_imp = pd.DataFrame(sorted(zip(load_clf.feature_importances_,input_df.columns)), 
                          columns=['Value','Feature']) 
            with _lock:
                fig = plt.figure(figsize=(20,15))
                sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))
                plt.ylabel('Feature Importance Score')
                st.pyplot(fig)

        



        # with st.beta_expander("Predict Flower Class"):
        #     s_len = st.number_input('Sepal Length')
        #     s_wid = st.number_input('Sepal Width')

        #     p_len = st.number_input('Petal Length')
        #     p_wid = st.number_input('Petal Width')
        #     model_name = 'models/decision_tree_model.sav'
        #     model = pickle.load(open(model_name,'rb'))
        #     if st.button('Perform the classification task'):

        #         mylist = np.array([s_len,s_wid,p_len,p_wid]).reshape(1,-1)

        #         result = interpreter(model.predict(mylist))
                
        #         result = "The species in question is"+" " + result
        #         st.title(result)
                




if __name__=="__main__":
    main()
