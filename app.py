import streamlit as st
from scraper import scrape_reviews
from visuals import create_histplot, create_lineplot, create_wordcloud, create_comparing_barplot, create_comparing_countplot, create_star_heatmap, create_comparing_lineplot
from ML import get_best_model, preprocess_data
import pandas as pd

st.header('Bewertungsanalyse')
st.markdown('##### Mit dieser App kannst du ganz einfach Bewertungen von von dir ausgewählten Unternehmen von Trustpilot laden und diese miteinander vergleichen. Außerdem kannst du mit den ausgewählten Bewertungen ein Modell trainieren, mit dessen Hilfe du anschließend für eigens eingegebene Rezensionen den Sternescore vorhersagen lassen kannst.')
st.sidebar.title('Parameter bestimmen')

company = st.sidebar.text_input('Die Bewertungen welches/welcher Unternehmens/App/Website willst du analysieren?', value='data-science-institute.de', help='Bitte achte darauf, den korrekten Namen der zugehörigen Website anzugeben.')

if 'reset_pagecount' not in st.session_state:
    st.session_state.reset_pagecount = False

reset_pagecount = st.sidebar.button('Alle vorhandenen Seiten scrapen', key='reset_pagecount_button')
if reset_pagecount:
    st.session_state.reset_pagecount = not st.session_state.reset_pagecount

if st.session_state.reset_pagecount:
    pagecount = None
else:
    pagecount = st.sidebar.slider('Wie viele Seiten Bewertungen möchtest du scrapen?', min_value=1, max_value=500, value=10, help='Je mehr Seiten du scrapest, desto präziser ist die Analyse - allerdings dauert sie dann auch länger.')

english = st.sidebar.selectbox('Möchtest du englischsprachige Bewertungen analysieren?', ['Ja', 'Nein'], index=1)
eng = english == 'Ja'

if 'scrape_button_clicked' not in st.session_state:
    st.session_state.scrape_button_clicked = False

def click_scrape():
    st.session_state.scrape_button_clicked = True

scrape = st.sidebar.button('Starte Scraping', on_click=click_scrape)

if st.session_state.scrape_button_clicked:
    new_df = scrape_reviews(company, pagecount, eng)

    if new_df is not None and not new_df.empty and 'company' in new_df.columns:
        if 'df' in st.session_state:
            st.session_state.df = pd.concat([st.session_state.df, new_df]).drop_duplicates().reset_index(drop=True)
        else:
            st.session_state.df = new_df

        if 'company' not in st.session_state:
            st.session_state.company = []
        st.session_state.company.append(company)

    else:
        st.error('Unbekannte Website.')

    st.session_state.scrape_button_clicked = False

df = st.session_state.get('df', pd.DataFrame())

if not df.empty and 'company' in df.columns:
    unique_companies = df['company'].unique()
    selected_companies = st.sidebar.multiselect('Wähle die Firmen zur Analyse aus:', options=unique_companies, default=unique_companies)

    filtered_df = df[df['company'].isin(selected_companies)]

    if not filtered_df.empty:
        if 'visuals' not in st.session_state:
            st.session_state.visuals = {}

        df_list = [filtered_df[filtered_df['company'] == key] for key in selected_companies]

        if 'histlist' not in st.session_state.visuals or len(st.session_state.visuals['histlist']) != len(df_list):
            st.session_state.visuals['histlist'] = [create_histplot(x) for x in df_list]
        if 'linelist' not in st.session_state.visuals or len(st.session_state.visuals['linelist']) != len(df_list):
            st.session_state.visuals['linelist'] = [create_lineplot(x) for x in df_list]
        if 'word_cloud_list' not in st.session_state.visuals or len(st.session_state.visuals['word_cloud_list']) != len(df_list):
            st.session_state.visuals['word_cloud_list'] = [create_wordcloud(x, eng, company) for x in df_list]

        comp_barplot = create_comparing_barplot(filtered_df)
        comp_countplot = create_comparing_countplot(filtered_df)
        comp_lineplot = create_comparing_lineplot(filtered_df)
        star_heatmap = create_star_heatmap(filtered_df)

        tab1, tab2, tab3 = st.tabs(["Visualisierungen", "Vergleichende Visualisierungen", "Daten und Sternepredictor"])

        with tab1:
            st.markdown('### Einzelvisualisierungen')
            st.write('')
            st.write('')
            for e in range(len(selected_companies)):
                st.markdown(f'#### Bewertungsübersicht für {selected_companies[e]}:')
                st.write('')
                st.write('Verteilung der Bewertungen:')
                st.pyplot(st.session_state.visuals['histlist'][e])
                st.write('')
                st.write('')
                st.write('Anzahl Bewertungen über die Zeit:')
                st.pyplot(st.session_state.visuals['linelist'][e])
                st.write('')
                st.write('')
                st.write('Am häufigsten verwendete Begriffe in den Bewertungen:')
                st.image(st.session_state.visuals['word_cloud_list'][e], use_column_width=True)
                st.write('')
                st.write('')
                st.write('')
                st.write('')

        with tab2:
            st.markdown('### Vergleichende Visualisierungen')
            st.write('Durchschnittliche Sternebewertungen:')
            st.pyplot(comp_barplot)
            st.write('')
            st.write('')
            st.write('Anzahl berücksichtigter Sternebewertungen:')
            st.pyplot(comp_countplot)
            st.write('')
            st.write('')
            st.write('Anzahl Bewertungen über die Zeit')
            st.pyplot(comp_lineplot)
            st.write('')
            st.write('')
            st.write('Verteilung der Sternebewertungen in Prozent:')
            st.pyplot(star_heatmap)
        
        with tab3:
            st.markdown('#### Dataframe')
            st.dataframe(filtered_df, height=300)
            st.write(f'Samplegröße: {len(filtered_df)}')
            st.markdown('#### Sternepredictor')

            if 'run_model_clicked' not in st.session_state:
                st.session_state.run_model_clicked = False

            def click_run_model():
                st.session_state.run_model_clicked = not st.session_state.run_model_clicked

            run_model = st.button('Model trainieren', on_click=click_run_model)

            if st.session_state.run_model_clicked:
                if len(filtered_df) < 50:
                    st.warning('Datenmenge zu gering für ein sinnvolles Modell!')
                else:
                    if len(filtered_df) < 1000:
                        st.warning('Sehr geringe Datenmenge - Ergebnisse können ungenau sein.')

                    X_transformed, y, vect = preprocess_data(filtered_df)
                    model, rmse = get_best_model(X_transformed, y)
                    st.session_state.model = model
                    st.session_state.rmse = rmse
                    st.session_state.vect = vect
                    st.session_state.df = filtered_df

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
