# Multi-label URL classification
This repo is about a multi-label classifier built with CNN, that takes an URL in input and predicts the list of labels.
To do that, we will use the URL structure. 

Obviously, we could have used the scraping methods. Why not use the **title** or **description** HTML tags to get much more information for our dataset ? 
We didn't do it for two reasons. In our corpus, we had anti-scrapping sites. So the first reason is more ethical.
The second reason is about the challenge. We preferred to try to make a model based only on the URLs.


## How to use

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
    "parsed_url": "jeuxvideo com wikis soluce astuces solution complete chapitre recherche et developpement ",
    "predictions": [
        [
            "1311",
            "381",
            "622",
            "925",
            "935"
        ]
    ],
    "inference_time": 0.068864
}
```

The documentation address is http://localhost:8000/docs

## Evaluation

For multi-label classification task, the **accuracy metrics** is not always a best way to evaluate the model.
Because, imagine that we have the following labels **[1311,381,622,925]** and the model predicts us **[381, 622, 925]**.
Taking into account these two lists of labels, the accuracy will be **0** whereas the model was able to predict good labels.

Contrary to that, the Jaccard index is used to understand the similarities between sample sets by calculating the size of the intersection divided by the size of the union of two label sets.
That's way, we preferred to use the Jaccard index to get a better evaluation metric.

| Model  | Accuracy  | Jaccard  |
| ------ |:---------:| ----:|
| CNN    | 0.13484    | 0.34194 |

The results are not good enough. There are two reasons for this. The first one is to have 1903 unique labels.
The second one is that URLs don't convey a lot of information sometimes.

 
