from django import forms

OPTIONS = (
    ("model_deployment.h5", "model_deployment.h5"),
    ("model7.h5","model7.h5"),
    ("model2.h5","model2.h5"),
    ("model1.h5","model1.h5"),
)
class FyreForm(forms.Form):
    image = forms.ImageField()
    latitude = forms.FloatField()
    longitude = forms.FloatField()

class InputForm(forms.Form):
    Soil_moisture = forms.FloatField()
    Soil_temp = forms.FloatField()
    Temp = forms.FloatField() 
    Wind = forms.FloatField()
    Humidity = forms.IntegerField()
    kmeansR = forms.IntegerField()
    kmeansG = forms.IntegerField()
    kmeansB = forms.IntegerField()
    model_name = forms.ChoiceField(widget=forms.RadioSelect,
                                         choices=OPTIONS)