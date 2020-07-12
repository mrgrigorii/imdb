import pickle

import keras.models

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from django.shortcuts import render
from django.utils import timezone


from .forms import ReviewForm
from .models import Review

model = keras.models.load_model('data/model_best.h5')
with open('data/tokenizer_word2vec.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
MAX_LEN = 231


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

            result = model.predict(x=tokens_pad).tolist()[0][0] * 10
            del tokens, tokens_pad, form

            new_review = Review(
                review_text=message,
                sub_date=timezone.now(),
                predicted_rating=result,
            )
            new_review.save()
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return render(request, 'reviews/review.html', {
                'form': ReviewForm(),
                'message_text': message,
                'result': 1 if result > 6 else 0,
                'score': round(result, 2),
            })

    # if a GET (or any other method) we'll create a blank form
    else:
        form = ReviewForm()

    return render(request, 'reviews/review.html', {'form': form})
