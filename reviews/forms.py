from django import forms


class ReviewForm(forms.Form):
    #subject = forms.CharField(max_length=100)
    message = forms.CharField(label='exampleTextarea', widget=forms.Textarea)
    #sender = forms.EmailField()
    #cc_myself = forms.BooleanField(required=False)
