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
import io
from io import BytesIO
from azure.storage.blob import BlockBlobService

tokenizer = nltk.RegexpTokenizer(r"\w+")
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from collections import OrderedDict
import yaml
import os
import warnings

warnings.filterwarnings('ignore')

std_desc_list = []
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

ACCNT_KEY = 'VNsOr8F7H9MwTcUOjtnKVv6LP5R6edR/4AMpLax2oe4pWqst44DSqm/SuiXto4sbb04E+1lIJKsJwFnDBFmh+Q=='


def load_conf():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + '/config.yaml', 'r')as yamlfile:
        return yaml.safe_load(yamlfile)


# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

parser = reqparse.RequestParser()
config = load_conf()


def new_matcher(row):
    te = dict()
    for s in std_desc_list:
        te.update({s: fuzz.token_set_ratio(row, s) - 30})
    # print(te)
    d = sorted(te.items(), key=lambda x: x[1], reverse=True)

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


def blob_reader(dir_name, file_name):
    file_name = file_name + '.xlsx'
    block_blob_service = BlockBlobService(account_name=config['CONTAINER_NAME'], account_key=config['ACCNT_KEY'])
    blob_item = block_blob_service.get_blob_to_bytes(dir_name, file_name)
    df = pd.read_excel(BytesIO(blob_item.content))
    return df


def blob_writer(df, dir_name, file_name):
    block_blob_service = BlockBlobService(account_name=config['CONTAINER_NAME'], account_key=config['ACCNT_KEY'])
    writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    block_blob_service.create_blob_from_path(dir_name, file_name, 'pandas_simple.xlsx')
    os.remove('pandas_simple.xlsx')


def data_load(des_f, att_f, cln_f, in_f):
    words_exc = blob_reader('Exclusion List')
    words_inc = blob_reader('Inclusion_List')
    dict_ref = blob_reader('Dictionary')


    df = pd.read_excel(r"C:\Users\shivam\Downloads\{}.xlsx".format(cln_f), sheet_name='cleaning_report')
    df2 = pd.read_excel(r"D:\AI_ML_WORK\Samples\{}.xlsx".format(in_f), sheet_name='Parts')

    std_desc = pd.read_excel(r"D:\AI_ML_WORK\Samples\{}.xlsx".format(des_f), sheet_name='PCDB_Oct_2021')

    std_attr = pd.read_excel(r"D:\AI_ML_WORK\Samples\{}.xlsx".format(att_f),
                             sheet_name='Padb_Oct_2021')

    return words_exc, words_inc, dict_ref, df, df2, std_desc, std_attr


def final_tax(des_f, att_f, cln_f, in_f):
    global std_desc_list
    words_exc, words_inc, dict_ref, df, df2, std_desc, std_attr = data_load(des_f, att_f, cln_f, in_f)
    context_words = words_exc.Word.to_list()
    context_words = list(set(context_words))
    context_words = [x for x in context_words if str(x) != 'nan']
    context_words = list(map(str.lower, context_words))

    incl_list = words_inc.Word.to_list()
    abbrev = dict(zip(dict_ref.Abbreviations.to_list(), dict_ref['Expansion  / Synonyms'].to_list()))
    abbr = dict((k.lower(), v.lower()) for k, v in abbrev.items())
    stop_words = stopwords.words('english')

    bt_df = std_desc[['PartTerminologyID', 'CategoryID']]
    bt_df.columns = ['Part Terminology ID', 'CategoryID']

    df2 = df2[['Part Number', 'Part Terminology ID']]
    df2.columns = ['PartNumber', 'Part Terminology ID']
    new_df = pd.merge(df, df2, how='left', on=['PartNumber'])
    new_df = pd.merge(new_df, bt_df, how='left', on=['Part Terminology ID'])
    new_df = new_df[new_df['type'] == 'description']

    std_desc = std_desc[std_desc['CategoryID'].isin(list(new_df.CategoryID.unique()))]
    std_desc['merge_cat'] = std_desc['PartTerminologyName']
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].lower(), axis=1)
    std_desc['merge_cat'] = std_desc['merge_cat'].apply(
        lambda row: ' '.join(OrderedDict((w, w) for w in row.split()).keys()))
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace(",", " "), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace(":", ""), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace("- ", "-"), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace(" -", "-"), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace('"', ''), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace(';', ''), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace(')', ''), axis=1)
    std_desc['merge_cat'] = std_desc.apply(lambda row: row['merge_cat'].replace('(', ''), axis=1)
    std_desc['tokens'] = std_desc.apply(lambda row: regexp_tokenize(row['merge_cat'], pattern=r"\s|[\.']", gaps=True),
                                        axis=1)
    std_desc['tail_trail_rem'] = std_desc['tokens'].apply(
        lambda row: [re.sub(r'[_+!@#$?^]+$', '', word) for word in row])
    std_desc['after_stopwords'] = std_desc['tail_trail_rem'].apply(
        lambda row: [word for word in row if word not in (stop_words)])
    std_desc['token_str'] = std_desc['after_stopwords'].apply(lambda x: ' '.join(map(str, x)))

    std_desc_list = list(set(std_desc.token_str.to_list()))

    new_df = new_df[new_df['After_Excluded_Numbers_Abbr_Symbol'] != '[]'].reset_index(drop=True)
    new_df['After_Excluded_Numbers_Abbr_Symbol'] = new_df['After_Excluded_Numbers_Abbr_Symbol'].apply(
        lambda row: eval(row))
    new_df['token_str'] = new_df['After_Excluded_Numbers_Abbr_Symbol'].apply(lambda x: ' '.join(map(str, x)))

    new_df.drop_duplicates('token_str', inplace=True)
    new_df['token_str'].replace('', np.nan, inplace=True)
    new_df.dropna(subset=['token_str'], inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    new_df['pred_attribute'] = new_df['token_str'].apply(
        lambda x: process.extract(x, std_desc_list, scorer=fuzz.token_sort_ratio))

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

    # new_df.drop(['after_spelling_check'], axis='columns', inplace=True)
    new_df.drop(['pred_attribute'], axis='columns', inplace=True)

    app_df = new_df[new_df['match_type'] == 'approximate'].reset_index(drop=True)
    exc_df = new_df[new_df['match_type'] == 'exact'].reset_index(drop=True)
    app_df['new_preds'] = app_df['token_str'].apply(new_matcher)

    app_df['recommended_pred'] = app_df['new_preds'].apply(lambda row: row[0] if len(row) > 0 else 'no_prediction')

    for i in range(1, 5):
        app_df['pred{}'.format(i)] = app_df['new_preds'].apply(lambda row: row[i] if len(row) > 0 else 'no_prediction')

    app_df.drop(['new_preds'], axis='columns', inplace=True)

    t_df = exc_df.append(app_df)

    ## final_taxonomy Predictions
    # ds_df
    def sep_attr(in_attr, std_attr):
        return list(set(std_attr) - set(in_attr))

    def attr_percentage(std, nott):
        if len(std) == 0:
            return 0
        return round(((len(std) - len(nott)) / len(std)) * 100, 2)

    def final_per_calc(des, attr):
        return round((des * 40 / 100) + (attr * 60 / 100), 2)

    std_attr['PAName'] = std_attr.apply(lambda row: row['PAName'].lower(), axis=1)

    te_df = t_df.copy(deep=True)
    re_df = te_df[['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']]
    pr1_df = te_df[['PartNumber', 'Original_Description', 'token_str', 'pred1']]
    pr2_df = te_df[['PartNumber', 'Original_Description', 'token_str', 'pred2']]
    pr3_df = te_df[['PartNumber', 'Original_Description', 'token_str', 'pred3']]
    pr4_df = te_df[['PartNumber', 'Original_Description', 'token_str', 'pred4']]

    re_df.columns = ['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']
    pr1_df.columns = ['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']
    pr2_df.columns = ['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']
    pr3_df.columns = ['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']
    pr4_df.columns = ['PartNumber', 'Original_Description', 'token_str', 'recommended_pred']

    re_df['type'] = 'recommend'
    pr1_df['type'] = 'pred1'
    pr2_df['type'] = 'pred2'
    pr3_df['type'] = 'pred3'
    pr4_df['type'] = 'pred4'

    ds_df = pd.concat([re_df, pr1_df, pr2_df, pr3_df, pr4_df], ignore_index=True)
    ds_df['ptid'] = ds_df['recommended_pred'].apply(
        lambda x: std_desc.loc[std_desc['token_str'] == x[0]]['PartTerminologyID'].values[0])
    ds_df = ds_df.sort_values(by=['PartNumber'], ascending=True).reset_index(drop=True)
    ds_df['des_confidence'] = ds_df['recommended_pred'].apply(lambda x: x[1])
    attr_df = df[df['type'] == 'attribute']
    ds_df['in_attributes'] = ds_df['PartNumber'].apply(
        lambda x: attr_df[attr_df['PartNumber'] == x].Original_Description.to_list())
    ds_df['std_attributes'] = ds_df['ptid'].apply(
        lambda x: std_attr[std_attr['PartTerminologyID'] == x].PAName.to_list())
    ds_df['std_attributes'] = ds_df['std_attributes'].apply(lambda row: list(set(row)))

    ds_df['not_attribute'] = ds_df.apply(lambda x: sep_attr(x.in_attributes, x.std_attributes), axis=1)
    ds_df['attr_confidence'] = ds_df.apply(lambda x: attr_percentage(x.std_attributes, x.not_attribute), axis=1)
    ds_df['tax_confidence'] = ds_df.apply(lambda x: final_per_calc(x.des_confidence, x.attr_confidence), axis=1)

    # print(ds_df)
    ds_df.to_excel(r"C:\Users\shivam\Downloads\final_tax_{}.xlsx".format(in_f), index=False)
    print('successfullly saved file')


# final_tax()

# exception_report(des_dt, att_dt)
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
class Final_tax(Resource):
    '''def get(self, file_n):
        return jsonify(exception_report(des_dt, att_dt,in_f,file_n))'''

    def post(self):
        parser.add_argument('des_f', type=str)
        parser.add_argument('att_f', type=str)
        parser.add_argument('cln_f', type=str)
        parser.add_argument('in_f', type=str)
        args = parser.parse_args()

        final_tax(des_f=args['des_f'], att_f=args['att_f'], cln_f=args['cln_f'],
                  in_f=args['in_f'])

        return {'Success': 'final taxonomy  {} is generated '.format(args['in_f'])}


# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Final_tax, '/final_tax')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
