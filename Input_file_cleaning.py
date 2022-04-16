import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import string
from nltk import regexp_tokenize
import re
import itertools
from textblob import TextBlob
import enchant
from itertools import chain
from nltk.tokenize import WhitespaceTokenizer
import pkg_resources
from symspellpy import SymSpell, Verbosity
from collections import OrderedDict
# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api,reqparse

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

parser = reqparse.RequestParser()

tk = WhitespaceTokenizer()


def data_loader(in_f):
    dict_ref = pd.read_excel(r"C:\Users\shivam\Downloads\Dictionary.xlsx", sheet_name='Sheet1')
    incl_wrd = pd.read_excel(r"C:\Users\shivam\Downloads\Inclusion_List.xlsx", sheet_name='Sheet1')
    words_exc = pd.read_excel(r"C:\Users\shivam\Downloads\Exclusion_List.xlsx", sheet_name='Sheet1')
    sp_mist = pd.read_excel(r"C:\Users\shivam\Downloads\Spelling_Mistakes.xlsx")
    stop_words = pd.read_excel(r"C:\Users\shivam\Downloads\Stop_Words.xlsx")
    des_dt = pd.read_excel(r"C:\Users\shivam\Downloads\{}.xlsx".format(in_f), sheet_name='Description')
    att_dt = pd.read_excel(r"C:\Users\shivam\Downloads\{}.xlsx".format(in_f), sheet_name='Attributes')
    # exc_rep = pd.read_excel(r"C:\Users\shivam\Downloads\base_taxonomy_spell_check (1)_new.xlsx")
    base_tax_words = pd.read_excel(r"C:\Users\shivam\Downloads\Base_Tax_Words.xlsx")
    std_attr_words = pd.read_excel(r"C:\Users\shivam\Downloads\Std_Attr_Words.xlsx")
    exc_rep = None

    abbrev = dict(zip(dict_ref.Abbreviations.to_list(), dict_ref['Expansion'].to_list()))
    abbr = dict((k.lower(), v.lower()) for k, v in abbrev.items())

    return dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr


#dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr = data_loader()

dict_ref = incl_wrd = sp_mist = stop_words = des_dt = att_dt = exc_rep = base_tax_words = std_attr_words = words_exc = abbr = 0

# After Exception  updating report
def exclusion_update(exc_rep1):
    global stop_words, incl_wrd, words_exc, dict_ref, sp_mist
    # Inclusions
    include = []
    for i in range(len(exc_rep1)):  # Finding the exceptions that are to be included
        if exc_rep1['Action'].iloc[i] == 'Include':
            include.append(exc_rep1['final_exceptions'].iloc[i])
    # Exclusions

    exclude = []
    for i in range(len(exc_rep1)):  # Finding the exceptions that are to  be excluded
        if exc_rep1['Action'].iloc[i] == 'Exclude':
            exclude.append(exc_rep1['final_exceptions'].iloc[i])

    # Spelling Mistakes
    spell_errors = []
    for i in range(len(exc_rep1)):  # Finding the exceptions that are identified as spelling mistakes
        if exc_rep1['Action'].iloc[i] == 'Spelling Mistake':
            spell_errors.append(exc_rep1['final_exceptions'].iloc[i])

    # Abbreviations
    abbreviations = []
    for i in range(len(exc_rep1)):  # Finding the exceptions that are identified as abbreviations
        if exc_rep1['Action'].iloc[i] == 'Abbrevation':
            abbreviations.append(exc_rep1['final_exceptions'].iloc[i])

    # Abbreviation Meanings
    meanings = []
    for i in range(len(exc_rep1)):  # Assigning expansions to the abbreviations
        if exc_rep1['Action'].iloc[i] == 'Abbrevation':
            meanings.append(exc_rep1['Corrected word'].iloc[i])

    # Making Inclusion List
    include = pd.DataFrame(include)
    include.rename(columns={0: 'Word'}, inplace=True)
    include_ = pd.concat([incl_wrd, include])
    include_.reset_index(drop=True, inplace=True)
    include_ = include_.drop_duplicates()
    include_.reset_index(drop=True, inplace=True)
    include_.to_excel(r"C:\Users\shivam\Downloads\Inclusion_List.xlsx", index=False)

    # Adding New Exclusions
    exclude = pd.DataFrame(exclude)
    exclude.rename(columns={0: 'Word'}, inplace=True)

    exclude['Exclusion Reason'] = ""
    for i in range(len(exclude)):
        exclude.iat[i, 1] = "Exclude"

    words_exc_ = pd.concat([words_exc, exclude])
    words_exc_.reset_index(drop=True, inplace=True)

    words_exc_ = words_exc_.drop_duplicates()
    words_exc_.reset_index(drop=True, inplace=True)

    words_exc_.to_excel(r"C:\Users\shivam\Downloads\Exclusion_List.xlsx", index=False)

    # Adding New Abbreviations
    abbreviations = pd.DataFrame(abbreviations)
    abbreviations.rename(columns={0: 'Abbreviations'}, inplace=True)

    meanings = pd.DataFrame(meanings)
    meanings.rename(columns={0: 'Expansion'}, inplace=True)

    abb = pd.concat([abbreviations, meanings], axis=1)

    dict_ref_ = pd.concat([dict_ref, abb])
    dict_ref_.reset_index(drop=True, inplace=True)

    dict_ref_ = dict_ref_.drop_duplicates()
    dict_ref_.reset_index(drop=True, inplace=True)

    dict_ref_.to_excel(r"C:\Users\shivam\Downloads\Dictionary.xlsx", index=False)

    # Spelling Mistakes List

    spell_errors = pd.DataFrame(spell_errors)
    spell_errors_ = pd.concat([sp_mist, spell_errors])
    spell_errors_.reset_index(drop=True, inplace=True)
    spell_errors_ = spell_errors_.drop_duplicates()
    spell_errors_.reset_index(drop=True, inplace=True)
    spell_errors_.to_excel(r"C:\Users\shivam\Downloads\Spelling_Mistakes.xlsx", index=False)

    ### Updating Stop Words List
    stop_words = list(stop_words.Word)
    stop_words.extend(words_exc_)

    # Removing those stop words which are present in standard attributes
    common_words_att = list(set(std_attr_words.Word) & set(stop_words))
    stop_words = list(set(stop_words) - set(common_words_att))

    # Removing those stop words which are present in Base taxonomy
    common_words_des = list(set(base_tax_words.Word) & set(stop_words))
    stop_words = list(set(stop_words) - set(common_words_des))
    stop_words = pd.DataFrame(stop_words, columns=['Word'])
    stop_words = stop_words.drop_duplicates()
    stop_words.reset_index(drop=True, inplace=True)
    # stop_words = pd.DataFrame(stop_words)
    stop_words.to_excel(r"C:\Users\shivam\Downloads\Stop_Words.xlsx", index=False)
    print('successfully Updates exceptions')


# cleaning functions
# Stops Dropper
def drop_stops(row):
    # Regex Pattern
    numbers = re.compile(r"[+-]?([0-9]*[.])?[0-9]+")
    alpha = re.compile(r"[a-zA-Z]")
    new_row = []
    for i in row:
        flag = 0
        if len(row) == 1:
            if len(i) == 1 or (len(i) > 1 and len(alpha.findall(i)) == 0):
                row[0] = ""
                return row
            else:
                check_num = numbers.findall(i)
                if i in abbr.keys() or i == "with" or i == "without":
                    new_row.append(i)
                else:
                    for stop in stop_words.Word:
                        if stop != i:
                            flag = flag + 1
                    if flag == len(stop_words):
                        new_row.append(i)
                return new_row
        else:
            check_num = numbers.findall(i)
            if i in abbr.keys() or i == "with" or i == "without":
                new_row.append(i)
            else:
                for stop in stop_words.Word:
                    if stop != i:
                        flag = flag + 1
                if flag == len(stop_words):
                    new_row.append(i)
    return new_row


# Numbers Dropper
def drop_nums(row):
    # Regex Pattern
    numbers = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    new_row = []
    for i in row:
        check_num = numbers.findall(i)
        if len(check_num) == 0:
            new_row.append(i)
    return new_row


# Symbol Dropper
def drop_syms(row):
    # Regex Patterns
    numbers = re.compile(r"[+-]?([0-9]*[.])?[0-9]+")
    with_ = re.compile(r"[w]{1}[/]{1}[a-zA-Z0-9][^o]{1}")
    with_out = re.compile(r"[w]{1}[/]{1}[o]{1}")
    hyphen_words = re.compile(r"[a-zA-Z0-9][-]+[a-zA-Z0-9]")
    new_row = []

    for i in row:
        flag = 0
        check_with = with_.findall(i)
        check_with_out = with_out.findall(i)

        if len(check_with) > 0:
            i = re.sub('w/', 'with ', i)
            for j in regexp_tokenize(i, pattern=r"[\s\$\&\+\,:;=?@#|\'\"<>\\/\^\*(\[\])%!\.`¾”“·™®°,‘–’]", gaps=True):
                new_row.append(j)
        elif len(check_with_out) > 0:
            i = "without"
            for j in regexp_tokenize(i, pattern=r"[\s\$\&\+\,:;=?@#|\'\"<>\\/\^\*(\[\])%!\.`¾”“·™®°,‘–’]", gaps=True):
                new_row.append(j)
            new_row.append(i)
        else:
            for a in dict_ref.Abbreviations.str.lower():
                if a != i:
                    flag = flag + 1

            if flag == len(dict_ref):
                check_num = numbers.findall(i)
                check_hyphen_words = hyphen_words.findall(i)
                if len(check_num) > 0:
                    new_row.append(i)
                elif len(check_hyphen_words) > 0:
                    for j in regexp_tokenize(i, pattern=r"[\$\&\+\,:;=?@#|\'\"<>\\/\^\*(\[\])%!\.`¾”“·™®°,‘–’]",
                                             gaps=True):
                        new_row.append(j)
                else:
                    for j in regexp_tokenize(i, pattern=r"[\$\&\+\,:;=?@#|\'\"<>\\/\^\*(\[\])%!\.`¾”“·™®°,‘–’-]",
                                             gaps=True):
                        new_row.append(j)
            else:
                new_row.append(i)
    return new_row


# Number Cleaner
def num_cleaner(row):
    # Regex Pattern
    numbers = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    new_row = []
    for i in row:
        check_num = numbers.findall(i)
        if len(check_num) > 0:
            for j in regexp_tokenize(i, pattern=r"[\$\&\+\,:;=?@#|\'\"<>\\/\^\*(\[\])%!\.`¾”“·™®°,‘–’-]",
                                     gaps=True):  # ask about backslash for division
                new_row.append(j)
        else:
            new_row.append(i)
    return new_row


# Spell Checker
def spell_chk(row):
    dict_1 = enchant.Dict("en_US")
    dict_2 = enchant.Dict("en_UK")
    new_row = []
    with_regex = re.compile(r"[a-zA-Z][with]")
    if len(row) != 0:
        for w in row:
            flag_1 = 0
            flag_2 = 0
            flag_3 = 0
            for incl in incl_wrd.Word.str.lower():

                if incl != w:
                    flag_1 = flag_1 + 1
            for spm in sp_mist.Word.str.lower():
                if spm != w:
                    flag_2 = flag_2 + 1
            for exp in dict_ref.Expansion.str.lower():
                if exp != w:
                    flag_3 = flag_3 + 1
            if w != "" and flag_1 == len(incl_wrd) and flag_2 == len(sp_mist) and flag_3 == len(dict_ref) and len(
                    with_regex.findall(w)) == 0:
                if not dict_1.check(w) and not dict_2.check(w):
                    new_row.append(w)
        return new_row
    else:
        return row


# attribute cleaning
def att_cleaning(supp_dt):
    supp_dt['Original_Description'] = supp_dt.apply(lambda row: row['AttributeHeaderName'].lower(), axis=1)
    supp_dt['Original_Tokens'] = supp_dt.apply(
        lambda row: regexp_tokenize(row['Original_Description'], pattern=r"\s", gaps=True), axis=1)
    supp_dt['After_Rem_Sym'] = supp_dt['Original_Tokens'].apply(drop_syms)
    supp_dt['After_Rem_Stops'] = supp_dt['After_Rem_Sym'].apply(drop_stops)
    supp_dt['After_Adding_Abbre'] = supp_dt['After_Rem_Stops'].apply(
        lambda row: [abbr[x] if x in list(abbr.keys()) else x for x in row])
    supp_dt['num_cleanup'] = supp_dt['After_Adding_Abbre'].apply(num_cleaner)
    supp_dt['After_Spell_Check'] = supp_dt['num_cleanup'].apply(spell_chk)

    supp_dt = supp_dt.drop(['AttributeHeaderName', 'PADBAttribute', 'num_cleanup'], axis=1)
    return supp_dt


# description_cleaning
def des_cleaning(supp_dt):
    supp_dt.columns = ['PartNumber', 'DescriptionCode', 'Original_Description']
    supp_dt = supp_dt[['PartNumber', 'Original_Description']]
    supp_dt = supp_dt.groupby(['PartNumber'])['Original_Description'].apply(' '.join).reset_index()
    supp_dt['Basic_cleanup_description'] = supp_dt.apply(lambda row: str(row['Original_Description']).lower(), axis=1)
    supp_dt['Original_Description'] = supp_dt['Original_Description'].apply(
        lambda row: ' '.join(OrderedDict((w, w) for w in row.split()).keys()))
    supp_dt['Original_Tokens'] = supp_dt.apply(
        lambda row: regexp_tokenize(row['Basic_cleanup_description'], pattern=r"\s", gaps=True), axis=1)
    supp_dt['After_Rem_Sym'] = supp_dt['Original_Tokens'].apply(drop_syms)
    supp_dt['After_Rem_Stops'] = supp_dt['After_Rem_Sym'].apply(drop_stops)
    supp_dt['After_Rem_Numbers'] = supp_dt['After_Rem_Stops'].apply(drop_nums)
    supp_dt['After_Adding_Abbre'] = supp_dt['After_Rem_Numbers'].apply(
        lambda row: [abbr[x] if x in list(abbr.keys()) else x for x in row])
    supp_dt['After_Spell_Check'] = supp_dt['After_Adding_Abbre'].apply(spell_chk)
    return supp_dt


def exception_report(in_f,exc_file=None):
    print('in_excpetion')
    global dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr
    if exc_file is not None:
        exc_rep1 = pd.read_excel(r"C:\Users\shivam\Downloads\{}.xlsx".format(exc_file), sheet_name='exception_report')
        print(exc_rep1.columns)
        dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr = data_loader(in_f)
        exclusion_update(exc_rep1)
        dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr = data_loader(in_f)

    elif exc_file is None:
        dict_ref, incl_wrd, sp_mist, stop_words, des_dt, att_dt, exc_rep, base_tax_words, std_attr_words, words_exc, abbr = data_loader(in_f)

    print('HERE')
    desc_df = des_cleaning(des_dt)
    attr_df = att_cleaning(att_dt)

    attr_df['Basic_cleanup_description'] = attr_df['Original_Description']
    attr_df['After_Rem_Numbers'] = "NA"

    attr_df['type'] = 'attribute'
    desc_df['type'] = 'description'

    new_df = pd.concat([desc_df, attr_df], ignore_index=True)

    new_df = new_df[['PartNumber', 'Original_Description', 'type', 'Basic_cleanup_description', 'Original_Tokens',
                     'After_Rem_Sym', 'After_Rem_Stops', 'After_Rem_Numbers', 'After_Adding_Abbre',
                     'After_Spell_Check']]

    new_df = new_df[new_df['After_Spell_Check'] != '[]'].reset_index(drop=True)
    new_df['After_Spell_Check'] = new_df['After_Spell_Check'].apply(lambda row: eval(str(row)))

    new_df.loc[~new_df['After_Spell_Check'].apply(bool), 'After_Spell_Check'] = ""

    res = pd.DataFrame(
        {'Original_Description': np.repeat(new_df['Original_Description'], new_df['After_Spell_Check'].map(len).values),
         'final_tokens': np.repeat(new_df['After_Adding_Abbre'], new_df['After_Spell_Check'].map(len).values),
         'final_exceptions': list(chain.from_iterable(new_df['After_Spell_Check'])),
         'type': np.repeat(new_df['type'], new_df['After_Spell_Check'].map(len).values)})

    res = res.drop_duplicates(['final_exceptions']).reset_index(drop=True)

    res = res.dropna(axis=0, how="any")

    writer = pd.ExcelWriter(r"C:\Users\shivam\Downloads\__Trial__5_1.xlsx", engine='xlsxwriter')

    # store your dataframes in a  dict, where the key is the sheet name you want

    frames = {'cleaning_report': new_df, 'exception_report': res}

    # now loop thru and put each on a specific sheet

    for sheet, frame in frames.items():  # .use .items for python 3.X
        frame.to_excel(writer, sheet_name=sheet)

    # critical last step
    writer.save()



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
        return jsonify({'message': 'Cleaning Pipeline API'})


# another resource to calculate the square of a number
class Cleaning(Resource):

    '''def get(self, file_n):
        return jsonify(exception_report(des_dt, att_dt,in_f,file_n))'''

    def post(self):
        parser.add_argument('inp_file', type=str)
        parser.add_argument('exc_file', type=str)

        args = parser.parse_args()

        print(args['exc_file'])
        if args['exc_file'] is None:
            exception_report(args['inp_file'])
            #return jsonify(exception_report( args['inp_file'], args['exc_file']))
            return {'Success': 'cleaning for i/p file {} is generated '.format(args['inp_file'])}
        else:
            exception_report(args['inp_file'], args['exc_file'])
            return {'Success': 'cleaning for i/p file {} is generated with exception report {} '.format(args['inp_file'],args['exc_file'])}



# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(Cleaning, '/clean')

# driver function
if __name__ == '__main__':
    app.run(debug=True)