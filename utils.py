import requests
import time
from datetime import datetime
import torch.nn.functional as F
import wandb


def download_weather_data():
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast?latitude=41.390205&longitude=2.154007&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability&forecast_days=1"
    )
    if response.status_code == 200:
        print("sucessfully fetched the data")
        return response.json()
    else:
        print(f"Hello person, there's a {response.status_code} error with your request")


def weather_data_generator():
    weather_data = download_weather_data()
    for measure in range(len(weather_data["hourly"]["time"])):
        hour = int(datetime.fromisoformat(weather_data["hourly"]["time"][measure]).hour)
        temperature = weather_data["hourly"]["temperature_2m"][measure]
        humidity = weather_data["hourly"]["relative_humidity_2m"][measure]
        apparent_temperature = weather_data["hourly"]["apparent_temperature"][measure]
        precipitation_probability = weather_data["hourly"]["precipitation_probability"][
            measure
        ]
        time.sleep(2)
        yield (
            hour,
            temperature,
            humidity,
            apparent_temperature,
            precipitation_probability,
        )


# convenience funtion to log predictions for a batch of test images
def log_test_predictions(
    images, labels, outputs, predicted, test_table, log_counter, NUM_IMAGES_PER_BATCH
):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(log_counter)
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break
