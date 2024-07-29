import streamlit as st
from scraper import scrape_reviews
from visuals import create_histplot, create_lineplot, create_wordcloud
from ML import get_best_model, preprocess_data
import pandas as pd

st.header('Bewertungsanalyse')
st.sidebar.title('Parameter bestimmen')

company = st.sidebar.text_input('Die Bewertungen welches/welcher Unternehmens/App/Website willst du analysieren?', value='data-science-institute.de', help='Bitte achte darauf, den korrekten Namen der zugehörigen Website anzugeben.')
pagecount = st.sidebar.slider('Wie viele Seiten Bewertungen möchtest du scrapen?', min_value=1, max_value=500, value=10, help='Je mehr Seiten du scrapest, desto präziser ist die Analyse - allerdings dauert sie dann auch länger.')
english = st.sidebar.selectbox('Möchtest du englischsprachige Bewertungen analysieren?', ['Ja', 'Nein'], index=1)
if english == 'Ja':
    eng = True
else:
    eng = False

if 'scrape_button_clicked' not in st.session_state:
    st.session_state.scrape_button_clicked = False
def click_scrape():
    st.session_state.scrape_button_clicked = not st.session_state.scrape_button_clicked
scrape = st.sidebar.button('Starte Scraping', on_click=click_scrape)

if st.session_state.scrape_button_clicked:
    new_df = scrape_reviews(company, pagecount, eng)
    if 'df' in st.session_state:
        st.session_state.df = pd.concat([st.session_state.df, new_df]).drop_duplicates().reset_index(drop=True)
    else:
        st.session_state.df = new_df
    if 'company' not in st.session_state:
        st.session_state.company = []
    if new_df is not None:
        st.session_state.company.append(company)
    st.session_state.pagecount = pagecount
    st.session_state.eng = eng

    df = st.session_state.get('df', pd.DataFrame())

    if df is not None:
        col1, col2 = st.columns([5, 2])
        hist = create_histplot(df)
        line = create_lineplot(df)
        word_cloud = create_wordcloud(df, eng, company)

        with col1:
            company_string = ', '.join(st.session_state.company)
            st.markdown(f'#### Bewertungsübersicht für {company_string}:')
            st.pyplot(hist)
            st.pyplot(line)
            st.image(word_cloud, use_column_width=True)


        with col2:
            st.markdown('#### Dataframe')
            st.dataframe(df, height=300)
            st.write(f'Samplegröße: {len(df)}')
            st.markdown('#### Sternepredictor')
            if 'run_model_clicked' not in st.session_state:
                st.session_state.run_model_clicked = False

            def click_run_model():
                st.session_state.run_model_clicked = not st.session_state.run_model_clicked

            run_model = st.button('Model trainieren', on_click=click_run_model)

            if st.session_state.run_model_clicked:
                if len(df) < 50:
                    st.warning('Datenmenge zu gering für ein sinnvolles Modell!')
                else:
                    if len(df) < 100:
                        st.warning('Sehr geringe Datenmenge - Ergebnisse können ungenau sein.')

                    # Train model if not already done
                    if 'model' not in st.session_state or 'vect' not in st.session_state or st.session_state.df is not df:
                        X_transformed, y, vect = preprocess_data(df)
                        model, rmse = get_best_model(X_transformed, y)
                        st.session_state.model = model
                        st.session_state.rmse = rmse
                        st.session_state.vect = vect
                        st.session_state.df = df
                    else:
                        model = st.session_state.model
                        rmse = st.session_state.rmse
                        vect = st.session_state.vect

                    st.markdown(f"Mittlere Abweichung des Models: {rmse:.2f} Sterne von der eigentlichen Bewertung")
                
                    comment = st.text_input('Zu prüfenden Kommentar eingeben:')
                
                    if comment:
                        try:
                            comment_tf = vect.transform([comment])
                            prediction = model.predict(comment_tf)
                            st.write(f'Dein Kommentar würde wahrscheinlich {prediction[0]} Sterne vergeben.')
                        except Exception as e:
                            st.error(f"Ein Fehler ist aufgetreten: {e}")

    else:
        st.markdown(f'# {company} wurde nicht gefunden!')
