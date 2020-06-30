import pickle

import keras.models

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from django.shortcuts import render


from .forms import ReviewForm

model = keras.models.load_model('data/model.h5')
with open('data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
MAX_LEN = 231

SCORE_DICT = {
    1: [0, 0.0003937620566782315],
    2: [0.0003937620566782315, 0.004110485660344286],
    3: [0.004110485660344286, 0.026892044431199115],
    4: [0.026892044431199115, 0.11231931946189017],
    5: [0.11231931946189017, 0.3085375387259869],
    6: [0.3085375387259869, 0.5848378711721023],
    7: [0.5848378711721023, 0.8234443827469972],
    8: [0.8234443827469972, 0.9497937530069486],
    9: [0.9497937530069486, 0.990791919711848],
    10: [0.990791919711848, 1]
}


def get_score(p):
    for key, interval in SCORE_DICT.items():
        if interval[0] < p < interval[1]:
            return key


def get_review(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = ReviewForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            message = form.cleaned_data['message']

            tokens = tokenizer.texts_to_sequences([message, ])
            tokens_pad = pad_sequences(tokens, maxlen=MAX_LEN)

            result = model.predict(x=tokens_pad).tolist()[0][0]
            score = get_score(result)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return render(request, 'reviews/review.html', {
                'form': form,
                'message_text': message,
                'result': int(round(result, 0)),
                'score': score,
            })

    # if a GET (or any other method) we'll create a blank form
    else:
        form = ReviewForm()

    return render(request, 'reviews/review.html', {'form': form})
