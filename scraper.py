from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from datetime import datetime



def scrape_reviews(company, pagecount, english = False):
    df = pd.DataFrame()
    if english:
        base_url = 'https://www.trustpilot.com/review/'
    else:
        base_url = 'https://de.trustpilot.com/review/'
    for i in range(1,pagecount+1):
        url = base_url+company+'?page='+str(i)
        response = requests.get(url)
        if response.status_code != 200:
            print('Keine weitere Seite gefunden.')
            break
        else:
            reviews = []
            ratings = []
            date = []
            soup = BeautifulSoup(response.content, "html.parser")
            script_tag = soup.find('script', type='application/ld+json', attrs={'data-business-unit-json-ld': 'true'})
            if script_tag:
                try:
                    json_data = json.loads(script_tag.string)
                except json.JSONDecodeError:
                    print("Fehler beim Decodieren der JSON-Daten.")
                else:
                    for element in (json_data['@graph']):
                        if element['@type'] == 'Review':
                            reviews.append(element['reviewBody'])
                            ratings.append(element['reviewRating']['ratingValue'])
                            datum_alt = datetime.strptime(element['datePublished'], "%Y-%m-%dT%H:%M:%S.%fZ")
                            datum_neu = datum_alt.strftime("%Y-%m-%d")
                            date.append(datum_neu)
                    review_df = pd.DataFrame({'reviews': reviews, 'ratings': ratings, 'date': date})
                    df = pd.concat([df, review_df], axis=0, ignore_index=True)
            else:
                print("Kein <script> Tag mit den gewünschten JSON-Daten gefunden.")
    return df