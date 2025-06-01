Start model with python 3.11


Create virtual environment with python 3.11
```
python -m venv .venv
```
or
```
ctrl/cmd + p, then type '>Python: Create Environment'
```


Activate venv,
On Windows:
```
.\.venv\Scripts\activate
```
On macOS/Linux:
```
source ./.venv/bin/activate
```


Install requirements
```
pip install -r requirements.txt
```


Run streamlit app
```
streamlit run app.py
```

Run api with fast api
```
uvicorn api:app --reload
```