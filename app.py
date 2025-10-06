from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import json
import time
import yfinance as yf
import os

app = FastAPI()
# Prefer Docker internal service; allow override via env
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")

# ===========================
# 🧠 SYSTEM PROMPT (ทุกโหมด)
# ===========================
SYSTEM_PROMPTS = {
    "default": (
        "คุณคือผู้ช่วยทางการเงินอัจฉริยะ (Financial Assistant) "
        "ที่สามารถวิเคราะห์และให้คำแนะนำด้านการเงิน การลงทุน และการบริหารหนี้สินในบริบทของประเทศไทยได้อย่างมืออาชีพ "
        "ตอบคำถามอย่างละเอียด ใช้เหตุผลเชิงวิเคราะห์ และสื่อสารด้วยภาษาที่เข้าใจง่าย เหมาะกับคนทั่วไป"
    ),
    "debt": (
        "คุณคือที่ปรึกษาทางการเงินที่เชี่ยวชาญด้านการบริหารหนี้สินในประเทศไทย "
        "ช่วยวิเคราะห์ภาระหนี้ อัตราดอกเบี้ย และให้แนวทางในการชำระหนี้อย่างมีประสิทธิภาพ "
        "แนะนำวิธีลดดอกเบี้ยและปรับโครงสร้างหนี้โดยไม่ทำให้คุณภาพชีวิตแย่ลง"
    ),
    "investment": (
        "คุณคือผู้เชี่ยวชาญด้านการลงทุนในประเทศไทยและต่างประเทศ "
        "ช่วยวิเคราะห์พอร์ตการลงทุนตามระดับความเสี่ยง คาดหวังผลตอบแทน และระยะเวลาการลงทุน "
        "สามารถแนะนำหุ้น กองทุน ตราสารหนี้ และกลยุทธ์การจัดพอร์ตให้เหมาะสมกับเป้าหมาย"
    ),
    "savings": (
        "คุณคือผู้เชี่ยวชาญด้านการวางแผนการออม "
        "ช่วยออกแบบแผนการออมเงินและลงทุนสำหรับเป้าหมายเฉพาะ เช่น ซื้อบ้าน เกษียณ หรือศึกษาต่อต่างประเทศ "
        "ตอบอย่างเป็นขั้นตอน ชัดเจน และเข้าใจง่าย"
    ),
    "tax": (
        "คุณคือผู้เชี่ยวชาญด้านภาษีและการวางแผนการเงินส่วนบุคคลในประเทศไทย "
        "ช่วยอธิบายวิธีลดหย่อนภาษี การวางแผนภาษีปลายปี และแนะนำกลยุทธ์ลงทุนเพื่อลดภาระภาษีอย่างถูกกฎหมาย"
    ),
}

# ===========================
# 🚀 Core Function
# ===========================
def stream_ollama(prompt: str, mode: str = "default"):
    """ส่ง prompt ไปยัง Ollama พร้อม system prompt"""
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["default"])

    # รวม system prompt เข้ากับ user prompt
    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    payload = {
        "model": "llama3.2",
        "prompt": full_prompt,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.2,
        "num_predict": 250
    }

    start_time = time.time()

    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
                except Exception as e:
                    yield f"[Error parsing line: {e}]"

    duration = time.time() - start_time
    print(f"✅ Ollama completed in {duration:.2f}s (mode: {mode})")


# ===========================
# 📡 FastAPI Routes
# ===========================
@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint รองรับทุกหมวด (debt/investment/savings/tax)"""
    data = await request.json()
    prompt = data.get("prompt", "")
    mode = data.get("mode", "default")  # 🧭 เพิ่ม parameter mode

    if not prompt:
        return JSONResponse({"error": "Missing 'prompt'"}, status_code=400)

    return StreamingResponse(stream_ollama(prompt, mode), media_type="text/plain")


@app.get("/finance/{symbol}")
async def get_stock(symbol: str):
    """ดึงข้อมูลราคาหุ้นจาก Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "symbol": symbol.upper(),
            "price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
            "short_name": info.get("shortName"),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Lightweight health check to confirm service and upstream URL
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
    }
