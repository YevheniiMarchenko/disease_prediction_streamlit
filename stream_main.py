import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def df_to_lists(dataframe):
    dna_text = list(dataframe['seq'])
    for item in range(len(dna_text)):
        dna_text[item] = ' '.join(dna_text[item])
    return dna_text


def occurrence_frequency(sequence, vectorizer='count'):
    if vectorizer.lower() == 'count':
        vectorizer = CountVectorizer(ngram_range=(1, 1), lowercase=False)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)

    fr_km = vectorizer.fit_transform(sequence)
    fr_km_array = fr_km.toarray()
    df_count = pd.DataFrame(data=fr_km_array, columns=vectorizer.get_feature_names_out())
    return df_count


def diagnosis_prediction(prediction):
    def diagnosis(seq):
        d_type = {"Healthy": 0, "Irritable bowel syndrome": 0, "T2D Prediabetes": 0}
        for n in d_type:
            d_type[n] = seq.count(n) / len(seq) * 100
        return d_type

    list_y = prediction.tolist()

    ndict = diagnosis(list_y)
    print(ndict)
    ndf = pd.DataFrame.from_dict(ndict, orient='index')
    ndf = ndf.reset_index()
    ndf = ndf.rename(columns={"index": "DIAGNOSIS", 0: "Probability"})

    fig_dia, ax1 = plt.subplots()
    plt.grid(True)
    ax1 = sns.barplot(x="DIAGNOSIS", y="Probability", data=ndf)
    ax1.set_title('Prediction')
    ax1.set_ylim(top=100, bottom=0)
    st.pyplot(fig_dia)


st.write("""
# Disease diagnosis
Prediction of 3 classes (*Healthy*, *Irritable bowel syndrome*, *T2D Prediabetes*)
""")

uploaded_file = st.file_uploader("Choose a file (only .pkl)")

list_4_mers = pd.read_pickle("datafile_4_mers_2100_per_class_3_classes.pkl")
df_4m = pd.DataFrame(list_4_mers)
dna_4m = df_to_lists(df_4m)
fr_4m = occurrence_frequency(dna_4m)
fr_4m_trg = pd.concat([fr_4m, df_4m[['class']]], axis=1)
st.dataframe(fr_4m_trg, 600, 200)

test_4m = pd.read_pickle(uploaded_file)
df_test = pd.DataFrame(test_4m)
df_test.drop('class', axis=1, inplace=True)
seq_test = df_to_lists(df_test)
fr_test = occurrence_frequency(seq_test)

x_train, x_test, y_train, y_test = train_test_split(fr_4m_trg.loc[:, fr_4m_trg.columns != 'class'],
                                                    fr_4m_trg.loc[:, 'class'].values, test_size=0.2)
tuned_parameters = [{'gamma': ['scale', 'auto', 2], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'tol': [1e-3, 1e-4]}]
clf = SVC(gamma='auto', kernel='rbf', tol=0.001)
clf.fit(x_train, y_train)

prediction = clf.predict(fr_test)

diagnosis_prediction(prediction)
plt.show()
