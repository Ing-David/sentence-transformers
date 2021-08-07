import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from urllib.parse import urljoin, urlparse
import requests
import random
from tqdm import tqdm
tqdm.pandas()

def uri_descripteurs(url_agritrop,email_address):
    """
    Extracts the URIs of global description via agritrop link web page
    :param url_agritrop:  URI of agritrop
    :param email_address: Email address uses to avoid blocking from agritrop's web page when trying to access many times
    """

    uri_descripteurs_agritrop = []
    # find id number in the url
    num = re.findall(r'\d+', url_agritrop)
    id_num = str(num[0])
    host = "http://agritrop.cirad.fr/cgi/export/eprint/"
    pattern = "/Simple/agritrop-eprint-"
    extension = ".txt"
    rel = id_num + pattern + id_num + extension
    # reconstruct the new url link automatically through the Metadata file text
    url_metadata = urljoin(host, rel)
    # request to url of publication
    headers = {
    'User-Agent': 'Mozilla/5.0',
    'From': email_address  # email address
    }

    response = requests.get(url_metadata,headers = headers)
    data = response.text
    for d in data.split("\n"):
        # find the pattern that cotains the word: 'agrovoc_mat_id:'
        pattern1 = re.compile(r'agrovoc_mat_id:')
        # domaine_name
        domain_name = "http://aims.fao.org/aos/agrovoc/"

        if (pattern1.findall(d)):
            key_word_1 = d.replace('agrovoc_mat_id: ', 'c_')
            url_key_word_1 = urljoin(domain_name, key_word_1)
            uri_descripteurs_agritrop.append(url_key_word_1)

    return uri_descripteurs_agritrop

def dataset_type(df,df_query,df_query_negative,type_dataframe):
    """
        Gets dataframe, it can be train, dev, or test set depend on variable type_dataframe
        :param df: Dataframe to execute
        :param df_query: Dataframe query with the column's name 'concept_ids'
        :param df_query_negative: Dataframe query with the column's name 'negative_concept'
        :param type_dataframe: It can be 'train', 'dev', or 'test'
    """
      # positive concepts
    df_positive = df.drop(columns='negative_concept')
    df_positive = df_positive.explode('concept_ids').reset_index(drop=True)
    df_positive_final = df_positive.merge(df_query, how = 'inner', on = ['concept_ids'])
    df_positive_final.insert(0, 'split', type_dataframe)
    df_positive_final.insert(1, 'score', 1)

    # negative concepts
    df_negative = df.drop(columns='concept_ids')
    df_negative = df_negative.explode('negative_concept').reset_index(drop=True)
    df_negative_final = df_negative.merge(df_query_negative, how = 'inner', on = ['negative_concept'])
    df_negative_final = df_negative_final.rename({'negative_concept': 'concept_ids'}, axis=1)
    df_negative_final.insert(0, 'split', type_dataframe)
    df_negative_final.insert(1, 'score', 0)

    # concatenate the dataframe of positive and negative concepts
    dataframe = pd.concat([df_positive_final,df_negative_final], ignore_index=True)
    dataframe["sentence2"] = dataframe.apply(lambda row: ",".join([x for x in row[["prefLabel", "altlabels"]] if not pd.isna(x)]),axis=1)
    dataframe = dataframe.drop(columns=['prefLabel','altlabels'])
    final_dataframe = dataframe.rename({'body_grobid': 'sentence1'}, axis=1)

    return final_dataframe

def get_dataframe(df, df_query, df_query_negative, all_classes, type_dataframe):

    """
    Gets random concepts for each type of dataframe
    :param df: Dataframe to execute
    :param df_query: Dataframe query with the column's name 'concept_ids'
    :param df_query_negative: Dataframe query with the column's name 'negative_concept'
    :param all_classes: all concepts in agrovoc
    :param type_dataframe: It can be 'train', 'dev', or 'test'
    """

    # Initialisation
    df['negative_concept'] = ''

    for i in range(0,len(df)):
        list_neg_can = []
        for a in range(0, 100):
            # random all concepts id in the entIdList
            m = random.sample(all_classes, 1)
            # avoid positive concepts
            if m[0] not in [j for j in df['concept_ids'][i]]:
                # avoid negative concepts that already existed in the list
                if m[0] not in [k for k in list_neg_can]:
                    list_neg_can.append(m[0])
                    # check if the random concepts are enough
                    if len(list_neg_can) == len(df['concept_ids'][i]):
                        break
        df['negative_concept'][i]= list_neg_can

    # list dataframe
    l = [None] * len(df)
    for i in range(0,len(df)):
        l[i] = dataset_type(df[i:i+1], df_query, df_query_negative, type_dataframe)

    dataframe = pd.concat(l, axis=0, ignore_index=True)

    return dataframe

def remove_uri_agrovoc(s):
    """
    Removes the domain name of agrovoc
    :param s: String to remove
    """
    s = re.sub("http://aims.fao.org/aos/agrovoc/","",s)
    return s

def remove_agritrop_link(s):
    """
    Removes the domain name of agritrop
    :param s: String to remove
    """
    s = re.sub("https://agritrop.cirad.fr/","",s)
    return s

def fichier_transformer_csv(original_csv, class_csv, query_csv, email_address, train_percent, test_percent,
                            abstract: bool, random_state):
    """
    Main function to get the overall dataframe (train + dev + test) ready to put into models
    :param original_csv: csv file of agritrop
    :param class_csv: csv file of all classes/concepts in agrovoc with their correspond URI
    :param query_csv: csv file of all classes/concepts with their altLabel and prefLabel
    :param email_address: Email address uses to avoid blocking from agritrop's web page when trying to access many times
    :param train_percent: put the percentage of train set eg. 0.6 mean 60%
    :param test_percent:  put the percentage of train set eg. 0.2 mean 20%. The rest i.e. 20% is for validation/dev
    :param abstract: types boolean in case we want an additional column for abstract's text
    :param random_state: random the dataframe with the same value
    """

    # read file csv orginal
    df_original = pd.read_csv(original_csv, index_col=0)
    # remove row that column of body_grobid contains null and NaN value
    df_clean = df_original[df_original['body_grobid'].isnull() == False]
    df_clean = df_original[df_original['body_grobid'].isna() == False]
    # filter column ACCES_AGRITROP and body_grobid
    df = df_clean[['ACCES_AGRITROP', 'body_grobid']]
    # dataset of all classes in agrovoc
    df_class = pd.read_csv(class_csv)
    # dataframe of concepts with their altLabel and prefLabel
    df_query = pd.read_csv(query_csv)
    df_query_negative = df_query.rename({'concept_ids': 'negative_concept'}, axis=1)

    list_concept_id = []
    for line, column in tqdm(df.iterrows()):
        # find URIs of each row of dataset
        uri_desc = uri_descripteurs(column['ACCES_AGRITROP'], email_address)
        # add it to the list
        list_concept_id.append(uri_desc)
    list_column = []
    for i in list_concept_id:
        list_row = []
        for j in i:
            # add it to the list
            list_row.append(j)
        list_column.append(list_row)
    # create dataframe with list of concept ids for each row
    df_concept_ids = pd.DataFrame()
    df_concept_ids['concept_ids'] = list_column
    df_text = df['body_grobid']
    # concatenate those two dataframes
    dataframe = pd.concat([df_text.reset_index(drop=True), df_concept_ids.reset_index(drop=True)], axis=1)
    # remove the row when the column concept_ids contain empty list
    dataframe_alter = dataframe[dataframe['concept_ids'].astype(bool)]
    # split the dataframe to train, test, val
    df_train, df_dev, df_test = np.split(dataframe_alter.sample(frac=1, random_state=random_state),
                                         [int(train_percent * len(dataframe_alter)),
                                          int((1 - test_percent) * len(dataframe_alter))])
    # reset index of train, test, val
    df_train_set = df_train.reset_index(drop=True)
    df_dev_set = df_dev.reset_index(drop=True)
    df_test_set = df_test.reset_index(drop=True)

    # all concepts in agrovoc
    all_classes = []
    for i in df_query['concept_ids']:
        all_classes.append(i)

    # dataframe for train, dev, and test
    dataframe_train = get_dataframe(df_train_set, df_query, df_query_negative, all_classes, type_dataframe='train')
    dataframe_dev = get_dataframe(df_dev_set, df_query, df_query_negative, all_classes, type_dataframe='dev')
    dataframe_test = get_dataframe(df_test_set, df_query, df_query_negative, all_classes, type_dataframe='test')

    # concatenate together
    dataframe_transformer = pd.concat([dataframe_train, dataframe_dev, dataframe_test], ignore_index=True)
    df_identify = df.rename({'body_grobid': 'sentence1'}, axis=1)
    df_tranformer_merge = dataframe_transformer.merge(df_identify, how='inner', on=['sentence1'])

    if abstract == True:
        # filter column RESUM and body_grobid
        df_abstract = df_clean[['RESUM', 'body_grobid']]
        df_abstract = df_abstract.rename({'body_grobid': 'sentence1'}, axis=1)
        df_tranformer_merge = df_tranformer_merge.merge(df_abstract, how='inner', on=['sentence1'])
        dataframe_final_transformer = df_tranformer_merge[
            ['split', 'score', 'RESUM', 'sentence1', 'sentence2', 'concept_ids', 'ACCES_AGRITROP']]
        dataframe_final_transformer = dataframe_final_transformer.rename(
            {'RESUM': 'abstract', 'ACCES_AGRITROP': 'doc_ids'}, axis=1)

    else:
        dataframe_final_transformer = df_tranformer_merge[
            ['split', 'score', 'sentence1', 'sentence2', 'concept_ids', 'ACCES_AGRITROP']]
        dataframe_final_transformer = dataframe_final_transformer.rename({'ACCES_AGRITROP': 'doc_ids'}, axis=1)

    # remove domaine name
    dataframe_final_transformer['concept_ids'] = dataframe_final_transformer['concept_ids'].progress_apply(
        remove_uri_agrovoc)
    dataframe_final_transformer['doc_ids'] = dataframe_final_transformer['doc_ids'].progress_apply(remove_agritrop_link)

    return dataframe_final_transformer

# dataframe_transformer = fichier_transformer_csv("corpus_titres_abstracts_corps_eng_articles-type_1_2_1000_limit.csv","class.csv","Query.csv",'prenom.nom@gmail.com', train_percent=0.6, test_percent=0.2, abstract= True, random_state = 42)
# dataframe_transformer.to_csv("corpus_agritrop_transformers.tsv",sep="\t", index=False)

