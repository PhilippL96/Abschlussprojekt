import streamlit as st
from scraper import scrape_reviews
from visuals import create_histplot, create_lineplot, create_wordcloud
from ML import get_best_model, preprocess_data
import pandas as pd

st.header('Bewertungsanalyse')
st.sidebar.title('Parameter bestimmen')

company = st.sidebar.text_input('Die Bewertungen welches/welcher Unternehmens/App/Website willst du analysieren?', value='data-science-institute.de', help='Bitte achte darauf, den korrekten Namen der zugehörigen Website anzugeben.')

reset_pagecount = st.sidebar.button('Alle vorhandenen Seiten scrapen')
if 'reset_pagecount' not in st.session_state:
    st.session_state.reset_pagecount = False

if reset_pagecount:
    st.session_state.reset_pagecount = True
    st.session_state.pagecount = None

if not st.session_state.reset_pagecount:
    pagecount = st.sidebar.slider('Wie viele Seiten Bewertungen möchtest du scrapen?', min_value=1, max_value=500, value=10, help='Je mehr Seiten du scrapest, desto präziser ist die Analyse - allerdings dauert sie dann auch länger.')
else:
    pagecount = None

english = st.sidebar.selectbox('Möchtest du englischsprachige Bewertungen analysieren?', ['Ja', 'Nein'], index=1)
eng = english == 'Ja'

# Scrape Button und Zustand
if 'scrape_button_clicked' not in st.session_state:
    st.session_state.scrape_button_clicked = False

if 'last_company' not in st.session_state:
    st.session_state.last_company = ''

def click_scrape():
    st.session_state.scrape_button_clicked = True

scrape = st.sidebar.button('Starte Scraping', on_click=click_scrape)

# Scraping nur ausführen, wenn der Button gedrückt wurde
if st.session_state.scrape_button_clicked:
    # Scraping durchführen
    new_df = scrape_reviews(company, pagecount, eng)

    # Sicherstellen, dass der DataFrame gültig ist und die Spalte existiert
    if new_df is not None and not new_df.empty and 'company' in new_df.columns:
        if 'df' in st.session_state:
            st.session_state.df = pd.concat([st.session_state.df, new_df]).drop_duplicates().reset_index(drop=True)
        else:
            st.session_state.df = new_df

        if 'company' not in st.session_state:
            st.session_state.company = []
        st.session_state.company.append(company)

        st.session_state.pagecount = pagecount
        st.session_state.eng = eng
        st.session_state.last_company = company  # Aktualisiere den zuletzt gescrapten Firmennamen

        # Generierung der Visualisierungen
        df_list = [st.session_state.df[st.session_state.df['company'] == key] for key in st.session_state.df['company'].unique()]
        st.session_state.histlist = [create_histplot(x) for x in df_list]
        st.session_state.linelist = [create_lineplot(x) for x in df_list]
        st.session_state.word_cloud_list = [create_wordcloud(x, st.session_state.eng, company) for x, company in zip(df_list, st.session_state.df['company'].unique())]
        st.session_state.visuals_generated = True

    else:
        st.error('Unbekannte Website.')

    st.session_state.scrape_button_clicked = False  # Setze den Button-Status zurück

# Überprüfen, ob der DataFrame gültig ist und die Spalte existiert
df = st.session_state.get('df', pd.DataFrame())
if not df.empty and 'company' in df.columns:
    col1, col2 = st.columns([5, 2])

    with col1:
        if 'visuals_generated' in st.session_state:
            for e in range(len(st.session_state.company)):
                st.markdown(f'#### Bewertungsübersicht für {st.session_state.company[e]}:')
                st.pyplot(st.session_state.histlist[e])
                st.pyplot(st.session_state.linelist[e])
                st.image(st.session_state.word_cloud_list[e], use_column_width=True)
        else:
            st.markdown('Keine Daten vorhanden oder Visualisierungen konnten nicht geladen werden.')

    with col2:
        st.markdown('#### Dataframe')
        st.dataframe(df, height=300)
        st.write(f'Samplegröße: {len(df)}')
        st.markdown('#### Sternepredictor')

        if 'run_model_clicked' not in st.session_state:
            st.session_state.run_model_clicked = False

        def click_run_model():
            st.session_state.run_model_clicked = True

        run_model = st.button('Model trainieren', on_click=click_run_model)

        if st.session_state.run_model_clicked:
            if len(df) < 50:
                st.warning('Datenmenge zu gering für ein sinnvolles Modell!')
            else:
                if len(df) < 100:
                    st.warning('Sehr geringe Datenmenge - Ergebnisse können ungenau sein.')

                # Train model if not already done or if new data has been scraped
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
    st.markdown(f'# Noch keine Daten vorhanden!')
