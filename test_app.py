# Assuming your test file is named test_app.py

import json
from api.app import app  # Importing the app object from api/app.py

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/", json={"suffix":suffix})
    assert response.status_code == 200    
    assert response.get_json()['op'] == "Hello, World POST "+suffix

# Add the following test case for post predict
def test_post_predict():
    # Replace the following line with the actual data from your dataset
    sample_data = {
        "sample_0": 0,
        "sample_1": 1,
        "sample_2": 2,
        "sample_3": 3,
        "sample_4": 4,
        "sample_5": 5,
        "sample_6": 6,
        "sample_7": 7,
        "sample_8": 8,
        "sample_9": 9,
    }

    for suffix, expected_digit in sample_data.items():
        response = app.test_client().post("/", json={"suffix": suffix})
        assert response.status_code == 200
        assert response.get_json()["predicted_digit"] == expected_digit

