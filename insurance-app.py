# import libraries
import base64
import os
import uuid
import re

import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from PIL import Image 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


plt.style.use('seaborn-notebook')
sns.set(context='paper', font='monospace', font_scale=3)


def main():
    page = st.sidebar.selectbox('Choose a page',['About App','Prediction and Evaluation'])

    if page == 'About App':
        st.title('Analysis and Prediction of Health Insurance Subscription in Nigeria')
        image = Image.open('images/GB.png')
        st.image(image)
        
        st.markdown("""
		This app predicts whether an individual would take up a health insurance policy or not leveraging 
		a machine learning classification model. We would also investigate factors that most likely influence taking up a health 
		insurance policy by an individual using the trained model. 

		Data obtained from Individual Recode section of the 2018 Nigerian Demographic and
		 Health Survey [DHS](https://dhsprogram.com/data/dataset/Nigeria_Standard-DHS_2018.cfm) .
		""")

        st.markdown("## Meet the Data Scientists")

        col1,mid,col2 = st.beta_columns(3)
        with col1:
            st.image('images/ope.jpg',width=300)
            html = f"Opeyemi Idris <a href='https://github.com/hardcore05' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.image('images/shakir.jpg',width=300)
            html = f"Shakiru Muraina <a href='https://github.com/Debare' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        col1,mid,col2 = st.beta_columns(3)
        with col1:
            st.image('images/bolu.jpg',width=300)
            html = f"Boluwatife Adewale <a href='https://github.com/BBLinus' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            st.image('images/uthman.jpg',width=300)
            html = f"Uthman Allison <a href='https://github.com/alliwene' alt='GitHub'><img height='20' src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMdSURBVFiFxZfLb9RWFMZ/99oeG3uYyZuUlD7UBgkCFCEkqKoiUKUuaOgCNhXTXdUisULdsOMPQAJWgAQSbBqWrEAtWzY8Fm0kQkBKKzXNJEI8JqOJx7FnPL4sZnAChdhRQuZbXY/POd835557jq+ghQmlTCuoH0NwFMU2wGF1UUUwhmLEN41Lg0IEAAJgyvMGkPpN4ItVJn0XRonC4U22PS1a//zeGpLHInzT2CutoH6sDeQAO62g/rNEUWgDeROCgpjya3NAtk0SXNlGcoCsbCM5APpyHRoKHlZ9/pzzeRqEAGwwdXatt9jqWGhiefHElF9TaY3vVzwuFmf5z6+/9f3HlsHxD7vYnVu3+gJGnpS5OlMmyVgCP23s5If+fCoBqWrg9xcuV2bKdBsaX3XY6OL/eTak4OsOm7yucXlmllslN5WAxBoohw0uFEsAfJm3OfFRN9NByD9ejX5TRwAzQchmO8MHps7pf59zq+RyoVhib34deU1bmYAbz+bwGhEAQdTcgAFTZ8BccB20M/E6UE1bN4y48WyOQn/HkvETt+BOZT5eH+pdn2TO9725Bd/y/BKWKQVMt46arUm2OGZiwG2ORUaK13xXJMCPmilVtGZ3AoRo9goAv7UdKxLQrTeLaL4RMeHVEgM+qgY0VFNBj7F0AaYSsCNrxevzxVJckG+DG0bxiQHY7ljvtE0t4GBPc1Yd7svRUIofx6a5WJxlclE3/Nurcb5YojBe5HE1iH//LkXRJgrYnrXY12nzxwuXE5u6+czOcK/ikdUWXE0puP60ghsuZOdAl8NQiqJN1YqrjYhfJ57gSMnJT3rpMSTaom4YRIqDo5Px82bH5MznG7C15EabqhU7muTcYD8dhuTo2BTf/jUZV/qb+KbL4WxKcljGOLY1yalP+3jQ63O77CEXnUldCI705djf6bA1RdoXY1nj+H2g7V9EEkg3N98PKhLFeNvoBeMSwW9tE6AYeXU1uwvsXGP60Ypp7JGDQgRE4TAwupbkROHwkBC1+DQ/VCqTC+q/ICi0ruerfWFxETxAca1iGpeGhKgBvARKYRTDyS8igAAAAABJRU5ErkJggg=='></a>"
            st.markdown(html, unsafe_allow_html=True)


    if page == 'Prediction and Evaluation':
        st.title('Predicting Health Insurance Subscription')
        st.sidebar.header('User Input Features')

        # st.sidebar.markdown("""
        # [Example CSV input file](https://raw.githubusercontent.com/alliwene/gb-november-grp2-health-insurance/main/data/data_sample.csv)
        # """)

        def download_button(object_to_download, download_filename, button_text):
            """
            Generates a link to download the given object_to_download.

            Params:
            ------
            object_to_download:  The object to be downloaded.
            download_filename (str): filename and extension of file. e.g. mydata.csv
            button_text (str): Text to display on download button (e.g. 'click here to download file')

            Returns:
            -------
            (str): the anchor tag to download object_to_download

            """

            try:
                # some strings <-> bytes conversions necessary here
                b64 = base64.b64encode(object_to_download.encode()).decode()

            except AttributeError as e:
                b64 = base64.b64encode(object_to_download).decode()

            button_uuid = str(uuid.uuid4()).replace('-', '')
            button_id = re.sub('\d+', '', button_uuid)

            custom_css = f""" 
                <style>
                    #{button_id} {{
                        background-color: rgb(255, 255, 255);
                        color: rgb(38, 39, 48);
                        padding: 0.25em 0.38em;
                        position: relative;
                        text-decoration: none;
                        border-radius: 4px;
                        border-width: 1px;
                        border-style: solid;
                        border-color: rgb(230, 234, 241);
                        border-image: initial;

                    }} 
                    #{button_id}:hover {{
                        border-color: rgb(246, 51, 102);
                        color: rgb(246, 51, 102);
                    }}
                    #{button_id}:active {{
                        box-shadow: none;
                        background-color: rgb(246, 51, 102);
                        color: white;
                        }}
                </style> """

            dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

            return dl_link


        def file_selector(folder_path='data'):
            filenames = os.listdir(folder_path)
            selected_filename = 'data_sample.csv'
            return os.path.join(folder_path, selected_filename)

        filename = file_selector()

        # Load selected file
        with open(filename, 'rb') as f:
            s = f.read()

        download_button_str = download_button(s, filename, 'Download sample input CSV file')
        st.sidebar.markdown(download_button_str, unsafe_allow_html=True)

        # Load cleaned dataset
        insurance_clean = pd.read_csv('data/data_clean.csv')
        insurance = insurance_clean.drop(columns=['target'])

		# Collects user input features into dataframe
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)

            internet = st.sidebar.selectbox('Use of internet',('Yes, last 12 months', 'Never', 'Yes, before last 12 months'))
            bank_acount = st.sidebar.selectbox('Account in bank',('Yes', 'No'))
            attainment = st.sidebar.selectbox("Husband/partner's educational attainment", 
                ('Complete secondary', 'No education', 'Higher', 'Complete primary',
                    'Incomplete secondary', "Don't know", 'Incomplete primary'))
            internet_freq = st.sidebar.selectbox('Internet use frequency',('At least once a week', 'Almost every day', 'Not at all',
                'Less than once a week'))
            literacy = st.sidebar.selectbox('Literacy', ('Able to read whole sentence', 'Cannot read at all',
                'Able to read only parts of sentence', 'Blind/visually impaired', 'No card with required language'))
            wealth_index = st.sidebar.slider('Wealth index', insurance['Wealth index factor score for urban/rural (5 decimals)'].min(), 
                insurance['Wealth index factor score for urban/rural (5 decimals)'].max(),
                float(input_df['Wealth index factor score for urban/rural (5 decimals)'][0]))
            toilet = st.sidebar.selectbox('Type of toilet facility', 
                ('Flush to piped sewer system', 'Flush to septic tank',
                    'Flush to pit latrine', 'Pit latrine with slab',
                    'Not a dejure resident', 'Pit latrine without slab/open pit',
                    'No facility/bush/field', 'Ventilated Improved Pit latrine (VIP)',
                    'Flush to somewhere else', 'Bucket toilet', 'Other',
                    'Composting toilet', "Flush, don't know where",
                    'Hanging toilet/latrine'))
            medical_help = st.sidebar.selectbox('Getting money needed for treatment', ('Not a big problem', 'Big problem'))
            residence = st.sidebar.selectbox('Type of place of residence', ('Urban', 'Rural'))
            medical_visit = st.sidebar.selectbox('Visited health facility last 12 months', ('Yes', 'No'))
            tv_watch = st.sidebar.selectbox('Frequency of watching television',
                ('At least once a week', 'Not at all', 'Less than once a week'))
            edu_year = st.sidebar.selectbox('Highest year of education', 
                ('3.0', '4.0', '2.0', '6.0', '1.0',
                    'No years completed at level V106', '5.0', '8.0', '7.0'))
            
            data = {'Use of internet': internet,
                    'Has an account in a bank or other financial institution': bank_acount,
                    "Husband/partner's educational attainment": attainment,
                    'Frequency of using internet last month': internet_freq,
                    'Literacy': literacy,
                    'Wealth index factor score for urban/rural (5 decimals)': wealth_index,
                    'Type of toilet facility': toilet,
                    'Getting medical help for self: getting money needed for treatment': medical_help,
                    'Type of place of residence': residence,
                    'Visited health facility last 12 months': medical_visit,
                    'Frequency of watching television': tv_watch,
                    'Highest year of education': edu_year,
                    }


            # Replace some values in input_ using data
            for key, value in data.items():
                input_df[key] = value


            # Combines user input features with cleaned dataset
            # This will be useful for the encoding phase
            df = pd.concat([input_df,insurance],axis=0,ignore_index=True)

            @st.cache()
            # one hot encode categorical features 
            def one_hot_encode(df):
                # get categorical features of df
                cat_feat = df.select_dtypes(exclude = np.number).columns 
                for feat in cat_feat:
                    dummy = pd.get_dummies(df[feat], prefix=feat)
                    df = pd.concat([df,dummy], axis=1)
                    del df[feat]
                input_df = df[:1] # Selects only the first row (the user input data)
                # remove duplicate columns
                input_df = input_df.loc[:,~input_df.columns.duplicated()] 
                return input_df

            input_df = one_hot_encode(df)
            
            st.subheader('User Input features')
            st.write(input_df)
        
            # Reads in saved classification model
            load_clf = pickle.load(open('model/insurance_rf.pkl', 'rb'))

            # Apply model to make predictions
            prediction = load_clf.predict(input_df)
            prediction_proba = load_clf.predict_proba(input_df)

            st.subheader('Prediction')
            output = np.array(['No','Yes'])
            st.write(output[prediction])

            st.subheader('Prediction Probability')
            st.write(prediction_proba)

            # make feature importance plot 
            st.subheader('Feature Importance Plot')
            st.markdown('''
                The top $30$ factors that most likely influence taking up an health insurance policy 
                by an individual is plotted. Values of some of these factors would be moved
                around to investigate its effect on our prediction. 
                ''')
            feat_imp = pd.DataFrame(sorted(zip(load_clf.feature_importances_,input_df.columns)), 
                          columns=['Value','Feature']) 
            imp_data = feat_imp.sort_values(by="Value", ascending=False)
            
            with _lock:
                fig = plt.figure(figsize=(20,15))
                sns.barplot(x="Value", y="Feature", data=imp_data.iloc[:30])
                plt.ylabel('Feature Importance Score')
                st.pyplot(fig)

        # Displays the user input features
        if uploaded_file is not None:
            st.write(' ')
        else:
            st.subheader('User Input features')
            st.write('Awaiting CSV file to be uploaded...')   



if __name__=="__main__":
    main()
