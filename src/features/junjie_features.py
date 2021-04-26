import os
import pandas as pd
import datetime
from genderperformr import GenderPerformr
from agreementr import Agreementr
from politenessr import Politenessr
from supportr import Supportr
import enchant
import requests
import json
from googleapiclient import discovery
from enchant.checker import SpellChecker
from enchant.tokenize import get_tokenizer
from nltk.tokenize import word_tokenize
import nltk
import time
nltk.download('punkt')


def clean_text(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return ' '.join(words)


def extract_features(tlc):
    """extract features from the text

    Args:
        tlc (dict[str]): all the attributes of a tlc

    Returns:
        [dict]: a dictionary of features extracted
    """
    text = clean_text(tlc['body'])
    fields = dict()
    # add features here #
    fields['Top_comment_word_count'] = len(text.split(' '))
    fields['Top_comment_text'] = text

    # Extract time-based features
    def get_day_of_week(text):
        return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S').weekday() + 1

    def get_day_of_month(text):
        return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S').day

    def get_time_of_day(text):
        return datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S').hour
    time_local = time.localtime(tlc['created_utc'])
    time_local = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    fields['Top_comment_day'] = get_day_of_month(time_local)
    fields['Top_comment_day_of_week'] = get_day_of_week(time_local)
    fields['Top_comment_hour'] = get_time_of_day(time_local)

    # Extract gender value
    gp = GenderPerformr()
    probs, _ = gp.predict(tlc['author'])
    # Rescale it from [0,1] to [-1,1]
    fields['Top_comment_author_gender_value'] = 2 * probs - 1

    # Extract percentage of mispellings
    check = SpellChecker("en_US")
    tokenizer = get_tokenizer("en_US")
    # Prevent the denominator from 0
    def weird_division(n, d):
        return n / d if d else 0

    def get_mispellings_percentage(text):
        mispelling_count = 0
        total_count = 0
        if text == 'nan':
            return total_count
        else:
            check.set_text(text)
            for err in check:
                mispelling_count = mispelling_count + 1
            for w in tokenizer(text):
                total_count = total_count + 1
            value = weird_division(mispelling_count, total_count)
            return value
    fields['Top_comment_mispellings'] = get_mispellings_percentage(text)

    # Get politeness, agreement, support scores, and rescale them from [1,5] to [-1,1]
    ar = Agreementr()
    pr = Politenessr()
    sr = Supportr()
    fields['Top_comment_agreement_value'] = 0.5*float(ar.predict([text]))-1.5
    fields['Top_comment_politeness_value'] = 0.5*float(pr.predict([text]))-1.5
    fields['Top_comment_support_value'] = 0.5*float(sr.predict([text]))-1.5

    # Get toxicity scores
    KEY = "yourkey.txt" # os.getenv("GOOGLE_API_KEY")
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=KEY)

    def get_results(request_id, response, exception):
        toxicity_scores.append((request_id, response))

    toxicity_scores = []
    count = 0
    batch = service.new_batch_http_request(callback=get_results)
    analyze_request = {
        'comment': {'text': text},
        "requestedAttributes": {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "ATTACK_ON_COMMENTER": {}
        }
    }
    batch.add(service.comments().analyze(body=analyze_request), request_id=str(count))
    batch.execute()
    toxic_score = toxicity_scores[0][1]['attributeScores']['TOXICITY']['summaryScore']['value']
    attack_score = toxicity_scores[0][1]['attributeScores']['ATTACK_ON_COMMENTER']['summaryScore']['value']
    if toxic_score > 0.5:
        fields['Top_comment_untuned_toxicity'] = 1
    else:
        fields['Top_comment_untuned_toxicity'] = 0
    if toxic_score > 0.8 and attack_score > 0.5:
        fields['Top_comment_tuned_toxicity'] = 1
    else:
        fields['Top_comment_tuned_toxicity'] = 0
    # end of feature extractions #
    return fields


def close(istream):
    """call deconstructors if needed
    """
    # e.g. close files or disconnect apis
    istream.close()


def main():
    istream, df = init()
    data = ["what's the date today?"]
    features = map(extract_features, data)
    close(istream)


main()
