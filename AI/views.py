import csv
import json
from cgi import test
from pprint import pprint
from re import template

import cv2
import keras
import numpy as np
import pandas as pd
import PIL
import requests
import tensorflow as tf
from django.contrib import messages
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.views import View
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .forms import FyreForm


# Create your views here.
class AIClassificationView(View):
    template_name = "pages/home.html"

    def make_histogram(request, cluster):
        """
        Count the number of pixels in each cluster
        :param: KMeans cluster
        :return: numpy histogram
        """

        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype("float32")
        hist /= hist.sum()
        return hist

    def make_bar(request, height, width, color):
        """
        Create an image of a given color
        :param: height of the image
        :param: width of the image
        :param: BGR pixel values of the color
        :return: tuple of bar, rgb values
        """
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        return bar, (red, green, blue)

    def get(self, request, *args, **kwargs):
        form = FyreForm()
        context = {"form": form}
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        form = FyreForm(request.POST, request.FILES)
        if form.is_valid():
            lat = form.cleaned_data.get("latitude")  # string
            lon = form.cleaned_data.get("longitude")  # string
            API_key = "6f854b594a11c357a475d54b73a2889e"  # replace with new user OpenWeatherMap API key (one listed here has been deprecated)
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            Final_url = (
                base_url + "appid=" + API_key + "&lat=" + str(lat) + "&lon=" + str(lon)
            )
           
            weather_data = requests.get(Final_url).json()
           
            # weather data
            temp = "%.1f" % ((weather_data["main"]["temp"]) - 273)
            wind_speed = "%.1f" % ((weather_data["wind"]["speed"]) * 3.6)
            humidity = "%.1f" % (weather_data["main"]["humidity"])
            
            # New Soil DATA api with AMBEE

            url = "https://api.ambeedata.com/soil/latest/by-lat-lng"
            querystring = {"lat": str(lat), "lng": str(lon)}
            headers = {
                "x-api-key": "01c1ebf7e2f4a6570e354598de03092e91a80a9a464e8cd4286911bf14d923d4",
                "Content-type": "application/json",
            }

            response = requests.request(
                "GET", url, headers=headers, params=querystring
            ).json()
          
            soil_temp = response["data"][0]["soil_temperature"]
            soil_moist = response["data"][0]["soil_moisture"]

            #Kmeans data 

            imgname_1 = request.FILES["image"]
            file_name = "pic.jpg"
            file_name_2 = default_storage.save(file_name, imgname_1)
            file_url = default_storage.url(file_name_2)
            img = cv2.imread("fyrewatchai/" + file_url)
            height, width, _ = np.shape(img)

            # # reshape the image to be a simple list of RGB pixels
            image = img.reshape((height * width, 3))

            # # get the most common color
            num_clusters = 1
            clusters = KMeans(n_clusters=num_clusters)
            clusters.fit(image)

            # # count and group dominant colors
            histogram = self.make_histogram(clusters)
            # then sort them, most-common first
            combined = zip(histogram, clusters.cluster_centers_)
            combined = sorted(combined, key=lambda x: x[0], reverse=True)

            # output a graphic showing the colors in order
            bars = []
            hsv_values = []
            for index, rows in enumerate(combined):
                bar, rgb = self.make_bar(100, 100, rows[1])
                dominance_text = f"  RGB values: {rgb}"
                rgb_list = list(rgb)
            kmeansR = rgb_list[0]
            KmeansG = rgb_list[1]
            kmeansB = rgb_list[2]

            
            data_labels_row = "SoilMoisture, SoilTemperature, Temperature, Wind Speed, Humidity, kmeansr, kmeansg, kmeansb"

            # write data to csv file
            with open("weather_data.csv", mode="w") as file:
                writer = csv.writer(
                    file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows([data_labels_row])
                writer.writerow(
                    [
                        soil_moist,
                        soil_temp,
                        temp,
                        wind_speed,
                        humidity,
                        kmeansR,
                        KmeansG,
                        kmeansB,
                    ]
                )
                # writer.writerow([0.071, 3.33, 6.38, 11.18, 85, 220, 191, 188])

            # neural network
            dataset = pd.read_csv("weather_data.csv")
            dataset.head()
            X = dataset.iloc[:, 0:8]
            Y = dataset.iloc[:, 1]
            X.head()
            obj = StandardScaler()
            X = obj.fit_transform(X)
            model = keras.models.load_model("AI/model_deployment.h5")
            y_pred = model.predict(X)
            
            for prediction in y_pred:
                if prediction > 0.96:
                    status = "Danger: Risk of Fire"
                    messages.warning(
                        self.request, "High Risk Of Wild Fire | Action is advised"
                    )
                elif prediction < 0.96:
                    status = "Low Danger: Low Risk of Fire"
                    messages.success(self.request, "Everything is alright!")
                else:
                    status = "try something else"
                    messages.success(self.request, "something is up")
            context = {
                "form": form,
                "status": status,
            }

        else:
            status = "Your Form failed"
            context = {
                "form": form,
                "status": status,
            }
        return render(request, self.template_name, context)
