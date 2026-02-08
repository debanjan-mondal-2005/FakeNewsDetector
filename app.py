from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
from pydantic import BaseModel
import uvicorn

# load preprocessing function
from preprocess import preprocess_text

app = FastAPI(title="Fake News Detection API")

# templates + static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# load model + tfidf
tfidf = joblib.load(r"D:\Semester 4 LPU\Gen AI(Own NLP Project)\fake_news_api\models\tfidf_vectorizer.pkl")
model = joblib.load(r"D:\Semester 4 LPU\Gen AI(Own NLP Project)\fake_news_api\models\fake_news_model.pkl")

# request body schema
class NewsInput(BaseModel):
    text: str

# ---------- UI PAGE ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- API PREDICT ----------
@app.post("/predict")
async def predict_news(data: NewsInput):

    clean_text = preprocess_text(data.text)
    vec = tfidf.transform([clean_text])
    pred = model.predict(vec)[0]

    result = "Real News" if pred == 1 else "Fake News"

    return JSONResponse(
        content={
            "prediction": result
        },
        media_type="application/json"
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
