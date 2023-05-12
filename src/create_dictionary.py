from scrabzl import Word, Dictionary
import unicodedata


def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)


def no_special_chars(word):
    ret = "'" not in word
    ret = ret and ' ' not in word
    ret = ret and '.' not in word
    ret = ret and '-' not in word
    return ret


def create_dictionaries(dictionary_path, max_word_length, language):
    words = []
    with open(dictionary_path, 'r') as f:
        for word in f.readlines():
            word = strip_accents(word).upper().strip()
            if (
                len(word) > 1 and len(word) <= max_word_length and
                no_special_chars(word)
            ):
                words.append(Word(word))

    words = tuple(sorted(set(words)))
    dictionary = Dictionary(words)
    dictionary.dump(language=language)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create dictionaries.')
    parser.add_argument('dictionary_path', metavar='dictionary-path', type=str,
                        help='Path to a dictionary txt file containing one word per line')
    parser.add_argument('dictionary_name', metavar='dictionary-name', type=str,
                        help='Name of the dictionary')
    parser.add_argument('--max-word-length', type=int, default=7,
                        help='Maximum word length of the words in the dictionary (default: 7)')

    args = parser.parse_args()

    create_dictionaries(args.dictionary_path, args.max_word_length, args.dictionary_name)