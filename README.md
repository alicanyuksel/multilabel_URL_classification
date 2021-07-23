# Multi-label URL classification
This repo is about a multi-label classifier built with CNN, that takes an URL in input and predicts the list of labels.
To do that, we will use the URL structure. 

Obviously, we could have used the scraping methods. Why not use the **title** or **description** HTML tags to get much more information for our dataset ? 
We didn't do it for two reasons. In our corpus, we had anti-scrapping sites. So the first reason is more ethical.
The second reason is about the challenge. We preferred to try to make a model based only on the URLs.

The DL model is built with [Keras](https://keras.io/api/) and the API with [FastAPI](https://fastapi.tiangolo.com).

## How to use

1. Clone the repo `https://github.com/alicanyuksel/multilabel_URL_classification.git`

2. Create a virtual environment
`
python3 -m venv venv
`

3. Install the requirements
`
pip install -r requirements.txt
`

4. Train the model

```
python train.py --path [PATH_DATA]
```
     
## Config
You can find all the config parameters in the **config/config.py** file.
You can change all the training parameters like **BATCH_SIZE**, **N_EPOCHS** etc.


       
## Inference API
After training the model, you could start our API server for inference.

    uvicorn app.main:app --port 8000 --reload

**That's it!** Now go to http://localhost:8000 !

### How to use the API


Send a **post request** to API (http://localhost:8000/predict):

```
{
    "url": "http://www.jeuxvideo.com/wikis-soluce-astuces/solution-complete-chapitre-9-recherche-et-developpement/194453"
}
```

You will get the following response :

```
{
    "url": "http://www.jeuxvideo.com/wikis-soluce-astuces/solution-complete-chapitre-9-recherche-et-developpement/194453",
    "parsed_url": "jeuxvideo com wikis soluce astuces solution complete chapitre recherche et developpement",
    "predictions": [
        [
            "1311",
            "381",
            "622",
            "925"
        ]
    ],
    "inference_time": 0.053776
}
```

The documentation address is http://localhost:8000/docs

## Data preprocessing

- Each URL is lower-cased and split into two categories like domain name and URI with regex.
- The protocols like (https, http, www) are removed.
- The punctuation are removed in the URI and the domain name. 
- All digits are removed in the URI, not in the domain name.
- The token's length less than 2 are removed.
- No stemming was done.

Example :

The URL in input :

```
http://www.jeuxvideo.com/wikis-soluce-astuces/solution-complete-chapitre-9-recherche-et-developpement/194453
```

After preprocessing:

```
{
    'protocol': 'http',
    'domainname': 'jeuxvideo com',
    'uri': 'wikis soluce astuces solution complete pr chapitre recherche et developpement'
}
```


## Evaluation

For multi-label classification task, the **accuracy metrics** is not always a best way to evaluate the model.
Because, imagine that we have the following labels **[1311,381,622,925]** and the model predicts us **[381, 622, 925]**.
Taking into account these two lists of labels, the accuracy will be **0** whereas the model was able to predict good labels.

Contrary to that, the Jaccard index is used to understand the similarities between sample sets by calculating the size of the intersection divided by the size of the union of two label sets.
That's way, we preferred to use the Jaccard index to get a better evaluation metric.

To compare our model CNN, we built also another model with Bidirectional LSTM. However the CNN model gives us better results.

| Model  | Accuracy  | Jaccard  |
| ------ |:---------:| ----:|
| CNN    | 0.1369    | 0.3354 |
| LSTM (Bidirectional) | 0.0755 | 0.1376 |

In conclusion, the results are not good enough. There are two reasons for this. The first one is to have 1903 unique labels to predict.
(**Extreme Multi-label Classification**)

The second one is that URLs don't always convey a lot of information. That's the challenge !

## How to improve

- We could try the scrapping methods to make the dataset richer. 
Ex: get title or description meta-tag and insert them into the dataset
- Why not clean the dataset ? This means removing some labels appearing very few times in all the dataset and see if it gives something better. 