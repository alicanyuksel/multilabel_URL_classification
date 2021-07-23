from datetime import datetime
from fastapi import FastAPI
from .api.models import TextInput, TextOutput, Welcome
from .api.classifier import get_model, get_tokenizer, parse_url, get_text_vectors, get_multilabelbinarizer
from config.config import API_NAME, API_DESCRIPTION


app = FastAPI(
    title=API_NAME,
    description=API_DESCRIPTION)


@app.on_event('startup')
def load_model():
    print("Model uploading...")
    app.state.model = get_model()
    app.state.tokenizer = get_tokenizer()
    app.state.multilabelbinarizer = get_multilabelbinarizer()


@app.get("/", response_model=Welcome)
async def read_root():
    return {}


@app.post("/predict", response_model=TextOutput)
async def predict(text: TextInput):
    start_time = datetime.now()
    # Get model tokenizer and multilabel binarizer
    model = app.state.model
    tokenizer = app.state.tokenizer
    mlb = app.state.multilabelbinarizer

    # Parse the input URL
    parsed_url = parse_url(text.url)

    # Vectorize the text
    tokenized_text = get_text_vectors(tokenizer, text=parsed_url)

    # Get prediction
    pred = model.predict(tokenized_text)

    # Get prediction's classes
    y_pred = mlb.inverse_transform(pred >= 0.5)

    end_time = datetime.now()
    inference_time = end_time - start_time

    return {
        "url": text.url,
        "parsed_url": parsed_url,
        "predictions": y_pred,
        "inference_time": inference_time
    }