# F1 Race Prediction API

A FastAPI + Machine Learning project that predicts whether an F1 driver will finish on the podium using historical Formula 1 race data.

## Tech Stack
- Python
- FastAPI
- Pandas
- Scikit-learn

## Goal
Build a machine learning model and serve predictions through a FastAPI backend.

## Version 1 Prediction Target
Predict whether a driver finishes on the podium:
- 1 = Podium
- 0 = Not Podium

## API Endpoints

### `GET /`
Root health/status message.

**Response**
```json
{ "message": "F1 Race Prediction API is running." }
```

### `GET /health`
Liveness check, including whether the trained model was loaded successfully.

**Response**
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Predicts whether a driver finishes on the podium given their grid/qualifying position for a race.

**Request body**
| Field                  | Type   | Description                                |
|------------------------|--------|--------------------------------------------|
| `driver`               | string | Driver full name (e.g. `"Max Verstappen"`)  |
| `team`                 | string | Constructor/team name (e.g. `"Red Bull"`)   |
| `circuit_name`         | string | Circuit name (e.g. `"Monza"`)               |
| `grid_position`        | int    | Starting grid position                      |
| `qualifying_position`  | int    | Qualifying session position                 |

```json
{
  "driver": "Max Verstappen",
  "team": "Red Bull",
  "circuit_name": "Bahrain International Circuit",
  "grid_position": 1,
  "qualifying_position": 1
}
```

**Response**
| Field                | Type    | Description                                |
|----------------------|---------|--------------------------------------------|
| `driver`             | string  | Driver from the request                    |
| `podium_probability` | float   | Predicted probability of finishing podium  |
| `podium_prediction`  | boolean | `true` if `podium_probability >= 0.5`      |

```json
{
  "driver": "Max Verstappen",
  "podium_probability": 0.9453,
  "podium_prediction": true
}
```

If the model hasn't been loaded (e.g. `app/model/podium_model.joblib` is missing), this endpoint returns `503`.
