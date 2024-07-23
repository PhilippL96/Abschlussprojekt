import streamlit as st
from scraper import scrape_reviews
from visuals import create_histplot, create_lineplot


st.header('Bewertungsanalyse')

col1, col2 = st.columns([5, 2])

company = st.sidebar.text_input('Die Bewertungen welches/welcher Unternehmens/App/Website willst du analysieren?', value='data-science-institute.de', help='Bitte achte darauf, den korrekten Namen der zugehörigen Website anzugeben.')
pagecount = st.sidebar.slider('Wie viele Seiten Bewertungen möchtest du scrapen?', min_value=1, max_value=100, value=10, help='Je mehr Seiten du scrapest, desto präziser ist die Analyse - allerdings dauert sie dann auch länger.')
english = st.sidebar.selectbox('Möchtest du englischsprachige Bewertungen analysieren?', ['Ja', 'Nein'], index=1)
if english == 'Ja':
    eng = True
else:
    eng = False

df = scrape_reviews(company, pagecount, eng)
hist = create_histplot(df)
line = create_lineplot(df)

with col1:
    st.markdown(f'#### Bewertungsübersicht für {company}:')
    st.pyplot(create_histplot(df))
    st.pyplot(create_lineplot(df))
