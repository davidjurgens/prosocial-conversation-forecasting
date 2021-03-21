import collections
from re import M
import pycld2
import pandas as pd
import sys


def is_English(text, verbose=False):
    """check if the string is in English

    Args:
        string (str): the string to be tested
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        bool: return True if it is in English
    """
    thre = 0.8  # threshold for % of English
    _, _, details, vectors = pycld2.detect(text, returnVectors=True)
    languages, _, _, _ = list(zip(*details))

    # ok if only contains English
    if set(languages) == {'ENGLISH'}:
        return True

    if not len(vectors):
        return False
    # calculate the length of each language
    _, lengths, langs, _ = list(zip(*vectors))

    counts = collections.defaultdict(int)
    for length, lang in zip(lengths, langs):
        counts[lang] += length

    # calculate % of English
    total_length = sum(counts.values())
    percent_of_English = counts['ENGLISH'] / total_length
    
    if verbose:
        print(f'% of English: {percent_of_English}')
    if set(languages) == {'ENGLISH', 'Unknown'} and languages[0] == 'ENGLISH' and percent_of_English > thre:
        return True
    
    return False

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    df = pd.read_csv(input_path, sep='\t', verbose=True)
    ise = df['Top_comment_text'].apply(lambda x: is_English(x))
    english_df = df[ise]
    english_df.to_csv(output_path, sep='\t', header=True, index=False)
    non_english_df = df[~ise]
    non_english_df.to_csv("/shared/0/projects/prosocial/train.nonenglish.tsv", sep='\t', header=True, index=False)


main()
