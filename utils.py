import requests
import time
from datetime import datetime

def download_weather_data():
    response = requests.get("https://api.open-meteo.com/v1/forecast?latitude=41.390205&longitude=2.154007&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability&forecast_days=1")
    if response.status_code == 200:
        print("sucessfully fetched the data")
        return response.json()
    else:
        print(f"Hello person, there's a {response.status_code} error with your request")


def weather_data_generator():
    weather_data = download_weather_data()
    for measure in range(len(weather_data['hourly']['time'])):
        hour = int(datetime.fromisoformat(weather_data['hourly']['time'][measure]).hour)
        temperature = weather_data['hourly']['temperature_2m'][measure]
        humidity = weather_data['hourly']['relative_humidity_2m'][measure]
        apparent_temperature = weather_data['hourly']['apparent_temperature'][measure]
        precipitation_probability = weather_data['hourly']['precipitation_probability'][measure]
        time.sleep(2)
        yield hour, temperature, humidity, apparent_temperature, precipitation_probability