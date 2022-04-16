import numpy as np
from itertools import chain
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import string
from nltk import regexp_tokenize
import re
import itertools
from textblob import TextBlob
import enchant
from nltk.tokenize import WhitespaceTokenizer
import numpy as np
from itertools import chain

tokenizer = nltk.RegexpTokenizer(r"\w+")
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')


# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api,reqparse

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

parser = reqparse.RequestParser()

std_attr = 'o',
words_exc = 'o',
words_inc = 'o',
dict_ref = 'o',
cln_attr = 'o',
context_words = 'o',
abbr = 'o',
std_att_list = 'o',


## data loader
def data_loader(std_attr_f, ex_f, in_f, dict_f, cln_f):
    std_attr = pd.read_excel(r"D:\APA_Deployment\master\{}".format(std_attr_f), sheet_name='PADB_NOV')
    words_exc = pd.read_excel(r"D:\APA_Deployment\master\{}".format(ex_f), sheet_name='Sheet1')
    words_inc = pd.read_excel(r"D:\APA_Deployment\master\{}".format(in_f), sheet_name='Sheet1')
    dict_ref = pd.read_excel(r"D:\APA_Deployment\master\{}".format(dict_f), sheet_name='Sheet1')
    cln_attr = pd.read_excel(r"D:\APA_Deployment\cleaning_report\{}".format(cln_f), sheet_name='Sheet1')

    '''if feed_f != 'NONE':
        feed_attr = pd.read_excel(r"D:\APA_Deployment\cleaning_report\{}".format(feed_f), sheet_name='Sheet1')
    else:
        feed_attr = 'NONE'''

    context_words = words_exc.Word.to_list()
    context_words = list(set(context_words))
    context_words = [x for x in context_words if str(x) != 'nan']
    context_words = list(map(str.lower, context_words))

    incl_list = words_inc.Word.to_list()
    abbrev = dict(zip(dict_ref.Abbreviations.to_list(), dict_ref['Expansion  / Synonyms'].to_list()))
    abbr = dict((k.lower(), v.lower()) for k, v in abbrev.items())

    std_attr = std_attr.drop_duplicates(['PAName']).reset_index(drop=True)
    std_attr['PAName'] = std_attr.apply(lambda row: row['PAName'].lower(), axis=1)

    base_tax_tokens = std_attr.PAName
    base_tax_words = list(itertools.chain(*base_tax_tokens))
    base_tax_words = list(map(str.lower, base_tax_words))
    # std_attr_list = list(set(base_tax_words))
    std_att_list = list(set(std_attr.PAName.to_list()))

    return std_attr, words_exc, words_inc, dict_ref, cln_attr, context_words, abbr, std_att_list


def new_matcher(row):
    te = dict()
    for s in std_att_list:
        te.update({s: fuzz.token_set_ratio(row, s) - 30})
    # print(te)
    d = sorted(te.items(), key=lambda x: x[1], reverse=True)

    # d = sorted(te.items(), key=lambda x: print(x) , reverse=True)[0:10]
    # print(d)
    def sorter():
        new_l = []
        init = 0
        while init < 5:
            for j in row.split():
                res = [i for i in filter(lambda k: j in k[0], d)]
                if len(res) != 0 and len(res) > init:
                    new_l.append(res[init])
                init += 1
                if init == 5:
                    break
        return new_l

    new_l = sorter()
    if len(new_l) < 5:
        for e in d:
            if e not in new_l:
                new_l.append(e)
                if len(new_l) == 5:
                    break

    new_l.sort(key=lambda x: x[1], reverse=True)
    return new_l


## attribute matching
def attribute_matching(std_attr_f='ListofStandardAttributes_nov_2021.xlsx', ex_f='Exclusion List.xlsx',
                       in_f='Inclusion_List.xlsx', dict_f='Dictionary.xlsx', cln_f='Att_5.xlsx', feed_f=None):
    global std_attr, words_exc, words_inc, dict_ref, cln_attr, context_words, abbr, std_att_list, feed_attr
    std_attr, words_exc, words_inc, dict_ref, cln_attr, context_words, abbr, std_att_list = data_loader(std_attr_f,
                                                                                                        ex_f, in_f,
                                                                                                        dict_f, cln_f)

    if feed_f is None:
        cln_attr['after_removing_english_stopwords'] = cln_attr['after_removing_english_stopwords'].apply(
            lambda row: eval(row))
        new_df = cln_attr[cln_attr['after_removing_english_stopwords'] != '[]'].reset_index(drop=True)
        new_df = new_df[
            ['PartNumber', 'Original_attribute', 'PADBAttribute', 'Tokenized_attribute', 'after_removing_symbols',
             'after_removing_english_stopwords', 'after_spelling_check']]
        new_df.loc[~new_df['after_removing_english_stopwords'].apply(bool), 'after_removing_english_stopwords'] = {
            np.nan}
        res = pd.DataFrame({'Original_attribute': np.repeat(new_df['Original_attribute'],
                                                            new_df['after_removing_english_stopwords'].map(len).values),
                            'after_removing_english_stopwords': list(
                                chain.from_iterable(new_df['after_removing_english_stopwords']))})

        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [word for word in row if word not in (context_words)])
        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [abbr[x] if x in list(abbr.keys()) else x for x in row])
        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [lemmatizer.lemmatize(w) for w in row])

        new_df['token_str'] = new_df['after_removing_english_stopwords'].apply(lambda x: ' '.join(map(str, x)))

        new_df.drop_duplicates('token_str', inplace=True)
        new_df['token_str'].replace('', np.nan, inplace=True)
        new_df.dropna(subset=['token_str'], inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df['pred_attribute'] = new_df['token_str'].apply(lambda x: process.extract(x, std_att_list))

        new_df['recommended_pred'] = new_df['pred_attribute'].apply(lambda row: row[0])

        for i in range(1, 5):
            new_df['pred{}'.format(i)] = new_df['pred_attribute'].apply(lambda row: row[i])

        new_df['match_type'] = 'approximate'

        for index, row in new_df.iterrows():
            if row['recommended_pred'][1] == 100:
                row['pred1'] = 'no_recommendation'
                row['pred2'] = 'no_recommendation'
                row['pred3'] = 'no_recommendation'
                row['pred4'] = 'no_recommendation'
                row['match_type'] = 'exact'

        new_df.drop(['after_spelling_check'], axis='columns', inplace=True)
        new_df.drop(['pred_attribute'], axis='columns', inplace=True)

        app_df = new_df[new_df['match_type'] == 'approximate'].reset_index(drop=True)
        exc_df = new_df[new_df['match_type'] == 'exact'].reset_index(drop=True)
        app_df['new_preds'] = app_df['token_str'].apply(new_matcher)

        app_df['recommended_pred'] = app_df['new_preds'].apply(lambda row: row[0] if len(row) > 0 else 'no_prediction')

        for i in range(1, 5):
            app_df['pred{}'.format(i)] = app_df['new_preds'].apply(
                lambda row: row[i] if len(row) > 0 else 'no_prediction')

        app_df.drop(['new_preds'], axis='columns', inplace=True)

        t_df = exc_df.append(app_df)
        t_df.to_excel(r"C:\Users\shivam\Downloads\test_attr_pred_sam5.xlsx", index=False)

    else:
        attribute_feedback_loop(feed_f)
        custom_match = pd.read_excel(r"D:\APA_Deployment\master\custom_matching.xlsx")
        alias_match = pd.read_excel(r"D:\APA_Deployment\master\alias_matching.xlsx")

        std_attr = std_attr.drop_duplicates(['PAName']).reset_index(drop=True)
        std_attr['PAName'] = std_attr.apply(lambda row: row['PAName'].lower(), axis=1)
        # std_attr['PAName'] = std_attr.apply(lambda row: tokenizer.tokenize(row['PAName']), axis=1)
        # cln_attr['After_Spell_Check'] = cln_attr['After_Spell_Check'].apply(lambda row: eval(row))

        # cln_attr = cln_attr.loc[cln_attr["type"] == 'attribute']

        # nltk.download('wordnet')
        # cln_attr.head()

        base_tax_tokens = std_attr.PAName
        base_tax_words = list(itertools.chain(*base_tax_tokens))
        base_tax_words = list(map(str.lower, base_tax_words))
        # std_attr_list = list(set(base_tax_words))
        std_att_list = list(set(std_attr.PAName.to_list()))

        ### Made compatible with new naming convention
        cln_attr['after_removing_english_stopwords'] = cln_attr['after_removing_english_stopwords'].apply(
            lambda row: eval(row))
        new_df = cln_attr[cln_attr['after_removing_english_stopwords'] != '[]'].reset_index(drop=True)
        new_df = new_df[
            ['PartNumber', 'Original_attribute', 'PADBAttribute', 'Tokenized_attribute', 'after_removing_symbols',
             'after_removing_english_stopwords', 'after_spelling_check']]
        new_df.loc[~new_df['after_removing_english_stopwords'].apply(bool), 'after_removing_english_stopwords'] = {
            np.nan}
        res = pd.DataFrame({'Original_attribute': np.repeat(new_df['Original_attribute'],
                                                            new_df['after_removing_english_stopwords'].map(len).values),
                            'after_removing_english_stopwords': list(
                                chain.from_iterable(new_df['after_removing_english_stopwords']))})

        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [word for word in row if word not in (context_words)])
        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [abbr[x] if x in list(abbr.keys()) else x for x in row])
        new_df['after_removing_english_stopwords'] = new_df['after_removing_english_stopwords'].apply(
            lambda row: [lemmatizer.lemmatize(w) for w in row])

        new_df['token_str'] = new_df['after_removing_english_stopwords'].apply(lambda x: ' '.join(map(str, x)))

        new_df.drop_duplicates('token_str', inplace=True)
        new_df['token_str'].replace('', np.nan, inplace=True)
        new_df.dropna(subset=['token_str'], inplace=True)
        new_df.reset_index(drop=True, inplace=True)

        new_df['pred_attribute'] = new_df['token_str'].apply(lambda x: process.extract(x, std_att_list))

        new_df['recommended_pred'] = new_df['pred_attribute'].apply(lambda row: row[0])

        for i in range(1, 5):
            new_df['pred{}'.format(i)] = new_df['pred_attribute'].apply(lambda row: row[i])

        new_df['match_type'] = 'approximate'

        ### Ailas Matching saving the ailas list
        alias_match = alias_match.drop(columns='id')

        ### cleaning custom match saving the custom attribute list
        custom_match = custom_match.drop(columns='id')

        ### Matching and generating recommendations
        for index, row in new_df.iterrows():
            ### basic Exact Match case
            if row.recommended_pred[1] == 100:
                new_df.loc[index, 'pred1'] = 'no_recommendation'
                new_df.loc[index, 'pred2'] = 'no_recommendation'
                new_df.loc[index, 'pred3'] = 'no_recommendation'
                new_df.loc[index, 'pred4'] = 'no_recommendation'
                new_df.loc[index, 'match_type'] = 'exact'
            ### check alias
            for item in alias_match.iterrows():
                if row.token_str == item[1]['alias_name']:
                    new_df.loc[index, 'recommended_pred'] = item[1]['std_attribute']
                    new_df.loc[index, 'pred1'] = 'no_recommendation'
                    new_df.loc[index, 'pred2'] = 'no_recommendation'
                    new_df.loc[index, 'pred3'] = 'no_recommendation'
                    new_df.loc[index, 'pred4'] = 'no_recommendation'
                    new_df.loc[index, 'match_type'] = 'exact-alias'
            ### check custom
            for item in custom_match.iterrows():
                if row.token_str == item[1]['custom_attribute']:
                    new_df.loc[index, 'recommended_pred'] = item[1]['recommended_attribute']
                    new_df.loc[index, 'pred1'] = 'no_recommendation'
                    new_df.loc[index, 'pred2'] = 'no_recommendation'
                    new_df.loc[index, 'pred3'] = 'no_recommendation'
                    new_df.loc[index, 'pred4'] = 'no_recommendation'
                    new_df.loc[index, 'match_type'] = 'exact-custom'

        # new_df.drop(['After_Spell_Check'], axis='columns', inplace=True)
        # new_df.drop(['pred_attribute'], axis='columns', inplace=True)

        ### creating dataframes filtered by Match type
        app_df = new_df[new_df['match_type'] == 'approximate'].reset_index(drop=True)
        exc_df = new_df[new_df['match_type'] == 'exact'].reset_index(drop=True)
        alias_exc_df = new_df[new_df['match_type'] == 'exact-alias'].reset_index(drop=True)
        cust_exc_df = new_df[new_df['match_type'] == 'exact-custom'].reset_index(drop=True)

        app_df['new_preds'] = app_df['token_str'].apply(new_matcher)

        app_df['recommended_pred'] = app_df['new_preds'].apply(lambda row: row[0] if len(row) > 0 else 'no_prediction')

        for i in range(1, 5):
            app_df['pred{}'.format(i)] = app_df['new_preds'].apply(
                lambda row: row[i] if len(row) > 0 else 'no_prediction')

        app_df.drop(['new_preds'], axis='columns', inplace=True)
        ### concatenating all the dataframes into one
        t_df = pd.concat([exc_df, alias_exc_df, cust_exc_df, app_df], axis=0, ignore_index=True)
        # t_df = t_df.drop(labels=['After_Adding_Abbre', 'After_Rem_Numbers', 'Basic_cleanup_description'],axis='columns')
        t_df.to_excel(r"C:\Users\shivam\Downloads\test_attr_pred_sam5_after_feedback.xlsx", index=False)


def attribute_feedback_loop(feed_f):
    ### change the file names
    ### Loading relevent files
    att_match_feedback = pd.read_excel(r"D:\APA_Deployment\attribute_matching\{}".format(feed_f))
    # cust_df_exc = pd.read_excel("/home/ps/apa_taxanomy/check_lists_12012022/custom_matching.xlsx").drop(columns='id')
    # alias_df_exc = pd.read_excel("/home/ps/apa_taxanomy/check_lists_12012022/alias_matching.xlsx").drop(columns='id')
    cust_df_exc = pd.DataFrame()
    alias_df_exc = pd.DataFrame()

    ### creating empty dicts for new feedback
    cust_df = {"custom_attribute": [], "recommended_attribute": []}
    alias_df = {"alias_name": [], "std_attribute": []}

    ### Adding action items to their respective dicts while also mainting the prediction format
    for index, row in att_match_feedback.iterrows():
        ### Custom
        if row['Action'] == 'Custom':
            cust_df['custom_attribute'].append(row['token_str'])
            cust_df['recommended_attribute'].append(f"('{row['Reccomended attrribute']}', 100)")
        ### Action
        if row['Action'] == 'Alias':
            alias_df['alias_name'].append(row['token_str'])
            temp = row['Reccomended attrribute'].split("'")[1]
            alias_df['std_attribute'].append(f"('{temp}', 100)")

    ### converting the dicts to dataframes
    cust_df = pd.DataFrame(cust_df)
    alias_df = pd.DataFrame(alias_df)

    ### Appending new items to existing lists
    alias_df_exc = alias_df_exc.append(alias_df, ignore_index=True)
    cust_df_exc = cust_df_exc.append(cust_df, ignore_index=True)

    ### droping duplicate id columns
    cust_df_exc.index.name = "id"
    alias_df_exc.index.name = "id"

    # print(cust_df_exc)
    # print(alias_df_exc)

    ### Saving dataframes as Excel
    cust_df_exc.to_excel(r"D:\APA_Deployment\master\custom_matching.xlsx")
    alias_df_exc.to_excel(r"D:\APA_Deployment\master\alias_matching.xlsx")

    return 'yes'


#t_df = attribute_matching(std_attr_f='ListofStandardAttributes_nov_2021.xlsx', ex_f='Exclusion List.xlsx',in_f='Inclusion_List.xlsx', dict_f='Dictionary.xlsx', cln_f='Att_5.xlsx', feed_f=None)

#sample5_attribute_predicted.xlsx

#exception_report(des_dt, att_dt)
# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Hello(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        return jsonify({'message': 'Attribute Prediction API'})


# another resource to calculate the square of a number
class Matching(Resource):

    '''def get(self, file_n):
        return jsonify(exception_report(des_dt, att_dt,in_f,file_n))'''

    def post(self):
        parser.add_argument('std_attr_f', type=str)
        parser.add_argument('ex_f', type=str)
        parser.add_argument('in_f', type=str)
        parser.add_argument('dict_f', type=str)
        parser.add_argument('cln_f', type=str)
        parser.add_argument('feed_f', type=str)

        args = parser.parse_args()

        if args['feed_f'] is None:
            attribute_matching(std_attr_f=args['std_attr_f'], ex_f=args['ex_f'],
                               in_f=args['in_f'], dict_f=args['dict_f'], cln_f=args['cln_f'])

            return {'Success': 'attribute matching {} is generated '.format(args['cln_f'])}
        else:
            attribute_matching(std_attr_f=args['std_attr_f'], ex_f=args['ex_f'],
                               in_f=args['in_f'], dict_f=args['dict_f'], cln_f=args['cln_f'],feed_f=args['feed_f'])

            return {'Success': 'attribute matching {} is generated with matching report {} '.format(args['cln_f'],args['feed_f'])}



# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Matching, '/matching')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
