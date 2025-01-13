from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import requests
import openmeteo_requests
from retry_requests import retry
from datetime import datetime, timedelta
import pandas as pd
import logging
import traceback

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트엔드에서의 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "../model/random_forest_model.joblib"
try:
    model = load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

airport = {
    "ICN" : {
        'latitude': 37.4602,
        'longitude': 126.4407,
        'timezone': "Asia/Seoul"
    },
    "NRT" : {
        'latitude': 35.7720,
        'longitude': 140.3929,
        'timezone': "Asia/Tokyo"
    },
    "KTX" : {
        'latitude': 34.4349,
        'longitude': 135.2448,
        'timezone': "Asia/Tokyo"
    },
    "FUK" : {
        'latitude': 33.5869,
        'longitude': 130.4517,
        'timezone': "Asia/Tokyo"
    },
    "CTS" : {
        'latitude': 42.7752,
        'longitude': 141.6923,
        'timezone': "Asia/Tokyo"
    }
}

airline = {
    "제주항공": 0,
    "진에어": 1,
    "대한항공": 2,
    "아시아나항공": 3,
    "티웨이항공": 4,
    "에어서울": 5,
    "에어부산": 6,
    "이스타항공": 7,
    "피치항공": 8,
    "집에어": 9,
    "에어재팬": 10,
    "에어프레미아": 11,
    "에어로케이항공(주)": 12,
    "에티오피안항공": 13
}

class FeatureInput(BaseModel):
    date: str
    scheduled_time: str
    departure_airport: str
    arrival_airport: str
    airline: str


# 날씨 데이터 가져오기
def fetch_weather_data(place):
    session = requests.Session()
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    hourly_params = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
    params = {
        "latitude": airport[place]['latitude'],
        "longitude": airport[place]['longitude'],
        "hourly": hourly_params,
        "timezone": airport[place]['timezone'],
        "forecast_days": 16
    }

    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True)
            .tz_convert(airport[place]['timezone'])
            .tz_localize(None),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
            .tz_convert(airport[place]['timezone'])
            .tz_localize(None),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
}

    for i in range(len(hourly_params)):        
        hourly_data[hourly_params[i]] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(hourly_data)

def handle_missing_values(df):
    df = df.dropna(how="all")
    
    for col_name in ["temperature_2m", "relative_humidity_2m", "dew_point_2m"]:
        if col_name in df.columns:
            mean_value = df[col_name].mean()
            df[col_name] = df[col_name].fillna(mean_value)
    
    for col_name in ["precipitation", "rain", "snowfall", "snow_depth", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]:
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(0)

    return df


# 날짜가 동일한 하나의 열만 추출
def match_weather_date(date, scheduled_time, df):
    hour = scheduled_time.split(':')[0]
    minute = scheduled_time.split(':')[1]
    if int(minute) > 30 and int(minute) < 60:
        hour = str(int(hour) + 1)
        if hour == '24':
            hour = '00'
            date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        weather_date = datetime.strptime(f"{date} {hour}:00:00", "%Y-%m-%d %H:%M:%S")
    else:
        weather_date = datetime.strptime(f"{date} {hour}:00:00", "%Y-%m-%d %H:%M:%S")

    matched_rows = df[df['date'] == weather_date]

    return matched_rows


# 항공사 벡터 생성
def make_airline_vector(df, airline_name):
    tmp = {}

    for i in range(14):
        num = 1 if i == airline[airline_name] else 0
        tmp[f'airline_vector_{i}'] = num

    tmp_df = pd.DataFrame([tmp])
    result = pd.concat([tmp_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    return result


# API 엔드포인트 정의
@app.post("/predict/")
def predict(input_data: FeatureInput):
    departure_airport = input_data.departure_airport
    arrival_airport = input_data.arrival_airport
    date = input_data.date
    scheduled_time = input_data.scheduled_time
    airline_name = input_data.airline
    
    try:
        departure_df = fetch_weather_data(departure_airport)
        arrival_df = fetch_weather_data(arrival_airport)

        df = pd.concat([departure_df.add_prefix('departure__'), arrival_df.add_prefix('arrival__')], axis=1)
        
        duplicated_columns = df.columns[df.columns.duplicated()]
        logging.info("Duplicated columns: %s", duplicated_columns.tolist())
        
        df = handle_missing_values(df)
        df = df.rename(columns={'departure__date': 'date'})

        matched_row = match_weather_date(date, scheduled_time, df)

        if matched_row.empty:
            raise HTTPException(status_code=400, detail="No matching weather data found for the given time.")

        data = make_airline_vector(matched_row, airline_name)

        data = data.drop(columns=['date', 'arrival__date'])
        data = data[model.feature_names_in_]

        # 예측값 및 확률값 계산
        prediction = model.predict(data)[0]
        prediction_proba = model.predict_proba(data)[0].tolist()

        # 결과 반환
        return {
            "prediction": int(prediction),
            "probability": prediction_proba[1]
        }
    except Exception as e:
        logging.error("An error occurred: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")