# Abschlussprojekt: Streamlit App für Trustpilot Rezensionen

## Projektübersicht

Diese Streamlit App ermöglicht es Benutzern, Bewertungen von Unternehmen auf Trustpilot zu scrapen, Visualisierungen der Bewertungen anzuzeigen und in Bezug auf die eingegebenen Unternehmen eine Sternebewertung für einen eingegebenen Kommentar mittels Machine Learning zu ermitteln. Hierbei handelt es sich um mein Abschlussprojekt im Rahmen eines Data-Science-Kompaktkurses.

## Funktionen

- **Unternehmenssuche**: Benutzer können die Website eines Unternehmens eingeben, um dessen Bewertungen auf Trustpilot zu scrapen.
- **Visualisierungen**: Die App zeigt verschiedene Visualisierungen der Bewertungen an, beispielsweise die Verteilung der Sternebewertungen. Wenn mehrere Unternehmen gescraped wurden, können diese miteinander verglichen werden.
- **Bewertungsschätzung**: Benutzer können eine Bewertung oder einen Kommentar eingeben, und die App ermittelt eine Sternebewertung mithilfe eines Machine Learning Modells.

## Installation und Nutzung
- Installiere die benötigten Python-Abhängigkeiten mit `pip install -r requirements.txt`.
- Starte die Streamlit App mit dem Befehl `streamlit run app.py`.
