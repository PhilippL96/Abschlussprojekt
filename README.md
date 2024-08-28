# Abschlussprojekt: Streamlit App für Trustpilot Rezensionen

## Projektübersicht

Diese Streamlit App ermöglicht es Benutzern, Bewertungen von Unternehmen auf [Trustpilot](https://de.trustpilot.com/) zu scrapen, Visualisierungen der Bewertungen anzuzeigen und in Bezug auf die eingegebenen Unternehmen eine Sternebewertung für einen eingegebenen Kommentar mittels Machine Learning zu ermitteln. Hierbei handelt es sich um mein Abschlussprojekt im Rahmen eines Data-Science-Kompaktkurses.

## Funktionen

- **Unternehmenssuche**: Benutzer können die Website eines Unternehmens eingeben, um dessen Bewertungen auf Trustpilot zu scrapen.
- **Visualisierungen**: Die App zeigt verschiedene Visualisierungen der gescrapten Bewertungen an, beispielsweise die Verteilung der Sternebewertungen. Wenn mehrere Unternehmen gescraped wurden, können diese miteinander verglichen werden.
- **Bewertungsschätzung**: Benutzer können eine Bewertung oder einen Kommentar eingeben, und die App ermittelt eine Sternebewertung mithilfe eines Machine Learning Modells.

## Skripte

- **scraper.py**: Enthält die Funktion, um für ein gegebenes Unternehmen eine bestimmte Zahl von oder alle Seiten zu scrapen. Liefert die relevanten Informationen zu den Bewertungen in einem Dataframe zurück.
- **visuals.py**: Enthält alle Funktionen, die für die Visualisierungen in den Dashboards benötigt werden. Diese erhalten i.d.R. einen Dataframe und liefern ein Figure-Objekt zurück. Für die Wordcloud wird ein buf-Objekt zurückgegeben.
- **ML.py**: Entält Funktionen, um
    1. die Daten für das Machine Learning vorzubereiten: Nimmt Dataframe entgegen, liefert vektorisiertes X, y und einen trainierten Vectorizer zurück.
    2. das Modell zu trainieren: Nimmt X und y entgegen, liefert bestes Modell und RMSE als Fehlerindikator zurück.
- - **app.py**: Setzt die Funktionen logisch in einer Streamlit-App zusammen.

## Installation und Nutzung
- Installiere die benötigten Python-Abhängigkeiten mit `pip install -r requirements.txt`.
- Starte die Streamlit App mit dem Befehl `streamlit run app.py`.
