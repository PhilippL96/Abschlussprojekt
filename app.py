import streamlit as st
from scraper import scrape_reviews
from visuals import create_histplot, create_lineplot
from ML import get_best_model, preprocess_data


st.header('Bewertungsanalyse')


company = st.sidebar.text_input('Die Bewertungen welches/welcher Unternehmens/App/Website willst du analysieren?', value='data-science-institute.de', help='Bitte achte darauf, den korrekten Namen der zugehörigen Website anzugeben.')
pagecount = st.sidebar.slider('Wie viele Seiten Bewertungen möchtest du scrapen?', min_value=1, max_value=500, value=10, help='Je mehr Seiten du scrapest, desto präziser ist die Analyse - allerdings dauert sie dann auch länger.')
english = st.sidebar.selectbox('Möchtest du englischsprachige Bewertungen analysieren?', ['Ja', 'Nein'], index=1)
if english == 'Ja':
    eng = True
else:
    eng = False

df = scrape_reviews(company, pagecount, eng)

if df is not None:
    col1, col2 = st.columns([5, 2])
    hist = create_histplot(df)
    line = create_lineplot(df)

    with col1:
        st.markdown(f'#### Bewertungsübersicht für {company}:')
        st.pyplot(create_histplot(df))
        st.pyplot(create_lineplot(df))

    with col2:
        st.markdown(f'#### Sternepredictor')
        if len(df) < 50:
            st.warning('Datenmenge zu gering für ein sinnvolles Modell!')
        else:
            if len(df) < 100:
                st.warning('Sehr geringe Datenmenge - Ergebnisse können ungenau sein.')
            
            X_transformed, y, vect = preprocess_data(df)
            model, rmse = get_best_model(X_transformed, y)
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
