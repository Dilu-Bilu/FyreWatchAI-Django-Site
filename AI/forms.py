from django import forms


class FyreForm(forms.Form):
    image = forms.ImageField()
    latitude = forms.IntegerField()
    longitude = forms.IntegerField()