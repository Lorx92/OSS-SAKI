import numpy
import pandas
import sklearn.feature_extraction.text
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.preprocessing


def df_from_csv(filepath: str) -> pandas.DataFrame:
    # import csv data, do some simple cleanup steps
    dataframe = pandas.read_csv(
        filepath,
        sep=';',
        encoding='utf8',
        header=0,
        # ignore column 'Waehrung' as it is always 'EUR'
        # ignore column 'Valutadatum' as only one row varies from 'Buchungstag'
        # ignore column 'Auftragskonto' as it only has 2 distinct values and 20% of the rows have no value
        # ignore columns 'Kontonummer' and 'BLZ' to improve generalization of the model
        # => personal decision that model should ignore account numbers
        usecols=lambda name: name not in ['Waehrung', 'Valutadatum', 'Auftragskonto', 'Kontonummer', 'BLZ'],
        # remaining: Buchungstag, Buchungstext, Verwendungszweck, Beguenstigter/Zahlungspflichtiger, Betrag, label
        dayfirst=True,
        parse_dates=['Buchungstag'],
        index_col=0,
        converters={'Betrag': lambda s: pandas.to_numeric(s.replace(",", "."), errors='coerce')}
    )
    return dataframe


def check_no_nan(dataframe: pandas.DataFrame):
    # clarifying NaN assumptions (no NaN expected)
    for col in dataframe.columns.to_list():
        nan_count = dataframe[col].isna().sum()
        if nan_count > 0:
            raise ValueError('{} NaN count = {}'.format(col, nan_count))


df_raw = df_from_csv('data.csv')
check_no_nan(df_raw)

# ------------------- preprocessing --------------------

# Buchungstext: categorical -> encode text as ordinal or one-hot
buchungstext_ordinal_ndarray, buchungstext_category = df_raw['Buchungstext'].astype('category').factorize()
buchungstext_ordinal_df = pandas.DataFrame(data=buchungstext_ordinal_ndarray, columns=['Buchungstext_ordinal'])
buchungstext_onehot_df = pandas.get_dummies(df_raw['Buchungstext'], prefix='Buchungstext_')

# label: categorical, target -> encode text as ordinal
# (one-hot not supported by multi-class bayes classifiers)
label_encoder = sklearn.preprocessing.OrdinalEncoder(dtype=numpy.int)
label_ordinal = label_encoder.fit_transform(df_raw['label'].to_numpy().reshape(-1, 1))

# feature extraction for Buchungstag and Betrag
# has no positive impact on classifier performance -> not included in joint matrix
buchungstag_df = pandas.DataFrame()
buchungstag_df['TageSeitMontag'] = df_raw['Buchungstag'].dt.dayofweek
buchungstag_df['TageSeitMonatsanfang'] = numpy.divide(df_raw['Buchungstag'].dt.day - 1,
                                                      df_raw['Buchungstag'].dt.daysinmonth - 1)
betrag_df = pandas.DataFrame()
betrag_df['BetragPositiv'] = df_raw['Betrag'] >= 0.0
betrag_df['BetragCents00'] = numpy.round(df_raw['Betrag']) - df_raw['Betrag'] == 0.0
betrag_df['Betrag2Log10'] = numpy.round(numpy.log10(numpy.abs(df_raw['Betrag']) + 1.0))

# bag of words feature extraction for textual columns (via pandas -> scikit-learn -> pandas)
# str.replace: deletes ~50% of words (those with digits) without lowering score
text_series = df_raw['Verwendungszweck'].\
    str.replace(r'\bde\d+\b', '', case=False, regex=True).\
    str.cat(others=df_raw['Beguenstigter/Zahlungspflichtiger'], sep=' ').\
    str.replace(r'\b\w+\d+\w+\b', '', regex=True).\
    str.replace(r'\b\d+\w*\b', '', regex=True)
vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=3, binary=False)
# almost same performance for min_df=1/2/3 and for binary=True/False
# compare to sklearn.feature_extraction.text.TfidfVectorizer -> TfidfVectorizer slightly worse
bag_of_words_matrix = vectorizer.fit_transform(text_series.to_numpy())  # sparse matrix (csr_matrix)
bag_of_words_column_names = vectorizer.get_feature_names()
print('bag of words: {} entries'.format(len(bag_of_words_column_names)))
bag_of_words_df = pandas.DataFrame(data=bag_of_words_matrix.todense(), columns=bag_of_words_column_names)

# ------------------- end of preprocessing --------------------

# create joint matrix
dataframes_to_join = [buchungstext_onehot_df, bag_of_words_df]  # left out buchungstag_df, betrag_df
X_df: pandas.DataFrame = pandas.concat(dataframes_to_join, axis=1)
X_column_names = []
for df in dataframes_to_join:
    X_column_names.extend(df.columns)
X_df.columns = X_column_names
y = label_ordinal.ravel()  # 2D -> 1D

# model training and evaluation
n_splits = 3
n_repeats = 40  # score fluctuations across runs are satisfactory for 40+ repeats
crossval = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
classifier_mnb = sklearn.naive_bayes.MultinomialNB()  # compare to sklearn.naive_bayes.ComplementNB
metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = sklearn.model_selection.cross_validate(classifier_mnb, X_df, y=y, cv=crossval,
                                                return_train_score=True, scoring=metric_names)
print(('averaged model scores for {n_repeats}x repeated stratified {k}-fold crossvalidation:'
       ).format(n_repeats=n_repeats, k=n_splits))
print('(+/- for one standard deviation; macro means scores averaged over all classes)')
max_len_names = max([len(s) for s in metric_names])
for metric_name in metric_names:
    test_score = scores['test_' + metric_name]
    train_score = scores['train_' + metric_name]
    print(('{metric_name: <' + str(max_len_names) + '} test set {test_mean:.3f} +/- {test_dev:.3f} ; ' +
           'training set {train_mean:.3f} +/- {train_dev:.3f}'
           ).format(metric_name=metric_name, test_mean=test_score.mean(), test_dev=test_score.std(),
                    train_mean=train_score.mean(), train_dev=train_score.std()))

# jupyter notebook evaluation:
#
# index_set_names = ['testing', 'training']
# eval_df = pandas.DataFrame(
#     index=pandas.MultiIndex(
#         names=['metric', 'over set'],
#         levels=[metric_names, index_set_names],
#         codes=[[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]]
#     ),
#     columns=['mean', 'stddev'],
#     dtype=numpy.float64
# )
#
# for metric_name in metric_names:
#     for index_set_name in index_set_names:
#         score_key = index_set_name.rsplit('ing')[0] + '_' + metric_name
#         metric_data = scores[score_key]
#         eval_df.at[(metric_name, index_set_name), 'mean'] = numpy.mean(metric_data)
#         eval_df.at[(metric_name, index_set_name), 'stddev'] = numpy.std(metric_data)
