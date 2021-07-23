import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from utils import clean_and_parse_url
from config.config import (
    TRAINED_MODEL_OUTPUT_PATH,
    MODEL_NAME,
    TOKENIZER_OUTPUT_PATH,
    LABELBINARIZER_OUTPUT_PATH,
    MAX_LEN
)


def get_model():
    """
    To get model traiend.
    :return: model trained
    """
    return tf.keras.models.load_model(f"{TRAINED_MODEL_OUTPUT_PATH}/{MODEL_NAME}")


def get_tokenizer():
    """
    To get the tokenizer
    :return: tokenizer
    """
    with open(TOKENIZER_OUTPUT_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def get_text_vectors(tokenizer, text: str):
    """
    To convert text into sequences.
    :param tokenizer: tokenizer
    :param text: parsed url
    :return: padded vector
    """
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, padding='post', maxlen=MAX_LEN)

    return padded_text


def get_multilabelbinarizer():
    """
    To get MultiLabelBinarizer
    :return: multilabelbinarizer
    """
    with open(LABELBINARIZER_OUTPUT_PATH, 'rb') as handle:
        mlb = pickle.load(handle)

    return mlb


def parse_url(url: str) -> str:
    """
    To parse the input url
    :param url: input url
    :return: url parsed
    """
    url_parsed = clean_and_parse_url(url)

    return url_parsed

