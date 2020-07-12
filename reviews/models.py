import datetime

from django.db import models
from django.utils import timezone


class Review(models.Model):
    review_text = models.TextField()
    sub_date = models.DateTimeField('date submitted')
    predicted_rating = models.FloatField()

    def __str__(self):
        return self.review_text

    def was_published_recently(self):
        return self.sub_date >= timezone.now() - datetime.timedelta(days=1)
