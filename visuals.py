import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import io

def create_histplot(df):
    df_sorted = df.sort_values(by='ratings').reset_index(drop=True)
    plt.figure(figsize=(10,4))
    sns.histplot(x=df_sorted['ratings'], kde=False)
    plt.xlabel('Sternebewertung')
    plt.ylabel('Anzahl')
    plt.grid(True)

    plt.xticks([1, 2, 3, 4, 5])

    fig = plt.gcf()
    plt.close()
    return fig



def create_lineplot(df):
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    
    df_grouped = df.groupby(['date', 'ratings']).size().reset_index(name='count')
    
    df_sorted = df_grouped.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    plt.figure(figsize=(10,4))
    sns.lineplot(x='date', y='count', data=df_sorted)
    plt.xlabel('Datum')
    plt.ylabel('Anzahl der Bewertungen')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    plt.xticks(rotation=45)
    
    fig = plt.gcf()
    plt.close()
    return fig

def create_wordcloud(df, english, name):
    short_name = name.split('.')[0]
    data = df['reviews']
    str_data = data.str.cat(sep=' ')
    if english:
        additional_stopwords = {'would', 'could', 'should', 'also', 'us', 'one', 'two', 'may', 'might','must', 'much', 'many', 'every', 'everything', 'nothing', 'anything', 'something','get', 'got', 'gets', 'getting', 'like', 'even', 'always', 'never', 'still', 'often','yet', 'however', 'though', 'although', 'whether', 'thus', 'therefore', 'hence', 'whereas', 'further', 'furthermore', 'moreover', 'besides', 'otherwise', 'instead','meanwhile', 'nonetheless', 'nevertheless', 'already', 'just', 'just', 'since', 'until','unto', 'towards', 'upon', 'throughout', 'within', 'without', 'upon', 'among', 'along','behind', 'beneath', 'beside', 'beyond', 'despite', 'during', 'except', 'inside', 'outside','toward', 'through', 'across', 'amongst', 'between', 'above', 'below', 'under', 'over', 'before', 'after', 'near', 'close', 'adjacent', 'next', 'in', 'on', 'at', 'by', 'with', 'about','against', 'like', 'out', 'off', 'over', 'up', 'down', 'into', 'for', 'since', 'per', 'than','too', 'very', 'also', 'again', 'thus', 'more', 'most', 'only', 'here', 'there', 'when', 'where', 'why', 'how', 'which', 'because', 'therefore', 'wherever', 'whenever', 'nowhere', 'somewhere','anywhere', 'everywhere', 'now', 'then', 'once', 'already', 'yet', 'still', 'ever', 'never','always', 'sometimes', 'usually', 'often', 'frequently', 'rarely', 'seldom', 'occasionally','hardly', 'barely', 'scarcely', 'thence', 'hence', 'thus', 'otherwise', 'meanwhile', 'furthermore','besides', 'likewise', 'similarly', 'hence', 'altogether', 'yet', 'nonetheless', 'nevertheless'}
    else:
        additional_stopwords = {'aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere','anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf','aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das','daß','dass', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben','dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen','dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch','ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger','einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer','eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin','hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in','indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jener', 'jenem', 'jenen', 'jener','jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können', 'könnte','machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem','meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun','nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst','sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll','sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unser', 'unsere', 'unserem', 'unseren','unserer', 'unseres', 'unter', 'viel', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was','weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden','wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'während','wurde', 'würde', 'würden','zu', 'zum', 'zur', 'zwar', 'zwischen'}
    
    additional_stopwords.update([short_name, name])
    stopwords = STOPWORDS.union(additional_stopwords)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap='viridis',
        contour_color='steelblue',
        contour_width=1
    ).generate(str_data)
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf



def create_comparing_barplot(df):
    grouped_df = df.groupby('company', as_index=False)['ratings'].mean()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='company', y='ratings', data=grouped_df)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')
    plt.xlabel('Unternehmen/App/Website')
    plt.ylabel('Durchschnittsbewertung')
    plt.grid(True)

    fig = plt.gcf()
    plt.close()
    return fig

def create_comparing_countplot(df):
    grouped_df = df.groupby('company', as_index=False)['ratings'].count()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='company', y='ratings', data=grouped_df)

    plt.xlabel('Unternehmen/App/Website')
    plt.ylabel('Durchschnittsbewertung')
    plt.grid(True)

    fig = plt.gcf()
    plt.close()
    return fig

def create_star_heatmap(df):
    heatmap_data = df.groupby(['company', 'ratings']).size().unstack(fill_value=0)

    heatmap_data_percent = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    heatmap_data_percent = heatmap_data_percent.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(10, len(heatmap_data_percent) * 0.5))
    ax = sns.heatmap(heatmap_data_percent, annot=True, fmt='.1f', cmap='coolwarm', linewidths=0.5)

    plt.xlabel('Sternebewertungen')
    plt.ylabel('Website')
    plt.yticks(rotation=0)

    fig = plt.gcf()
    plt.close() 
    return fig


def create_comparing_lineplot(df):
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    
    df_grouped = df.groupby(['date', 'company']).size().reset_index(name='count')
    
    df_sorted = df_grouped.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    plt.figure(figsize=(10,4))
    sns.lineplot(x='date', y='count', data=df_sorted, hue='company')
    plt.xlabel('Datum')
    plt.ylabel('Anzahl der Bewertungen')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    plt.xticks(rotation=45)
    
    fig = plt.gcf()
    plt.close()
    return fig