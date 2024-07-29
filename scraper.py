import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from datetime import datetime

def scrape_reviews(company, pagecount=None, english=False):
    df = pd.DataFrame()
    base_url = 'https://de.trustpilot.com/review/' if not english else 'https://www.trustpilot.com/review/'
    company_url = base_url + company

    test_res = requests.get(company_url)
    if test_res.status_code != 200:
        return None

    def extract_reviews_from_page(url):
        response = requests.get(url)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        script_tag = soup.find('script', type='application/ld+json', attrs={'data-business-unit-json-ld': 'true'})
        if not script_tag:
            print("Kein <script> Tag mit den gewünschten JSON-Daten gefunden.")
            return None

        try:
            json_data = json.loads(script_tag.string)
        except json.JSONDecodeError:
            print("Fehler beim Decodieren der JSON-Daten.")
            return None

        reviews, ratings, dates = [], [], []
        for element in json_data['@graph']:
            if element['@type'] == 'Review':
                reviews.append(element['reviewBody'])
                ratings.append(element['reviewRating']['ratingValue'])
                datum_alt = datetime.strptime(element['datePublished'], "%Y-%m-%dT%H:%M:%S.%fZ")
                datum_neu = datum_alt.strftime("%Y-%m-%d")
                dates.append(datum_neu)

        review_df = pd.DataFrame({'reviews': reviews, 'ratings': ratings, 'date': dates})
        return review_df

    if pagecount is not None:
        for i in range(1, pagecount + 1):
            url = f'{company_url}?page={i}'
            review_df = extract_reviews_from_page(url)
            if review_df is None:
                print('Keine weitere Seite gefunden.')
                break
            df = pd.concat([df, review_df], axis=0, ignore_index=True)
    else:
        i = 1
        while True:
            url = f'{company_url}?page={i}'
            review_df = extract_reviews_from_page(url)
            if review_df is None:
                print('Keine weitere Seite gefunden.')
                break
            df = pd.concat([df, review_df], axis=0, ignore_index=True)
            i += 1

    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    df['company'] = company
    return df
