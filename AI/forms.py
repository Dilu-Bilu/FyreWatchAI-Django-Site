from django import forms


class FyreForm(forms.Form):
    image = forms.ImageField()
    latitude = forms.IntegerField()
    longitude = forms.IntegerField()

class InputForm(forms.Form):
    Soil_moisture = forms.FloatField()
    Soil_temp = forms.FloatField()
    Temp = forms.FloatField() 
    Wind = forms.FloatField()
    Humidity = forms.IntegerField()
    kmeansR = forms.IntegerField()
    kmeansG = forms.IntegerField()
    kmeansB = forms.IntegerField()