from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fanato Finance LLM API")

# Configuration
USE_TGI = os.getenv("USE_TGI", "false").lower() == "true"
USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"
TGI_BASE_URL = os.getenv("TGI_BASE_URL", "http://localhost:80")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "AdaptLLM/finance-chat")

# Thai financial system prompt
FINANCE_SYSTEM_PROMPT = """คุณคือผู้ช่วยทางการเงินส่วนบุคคลสำหรับคนไทย มีความเชี่ยวชาญด้าน:
- กฎหมายภาษีเงินได้บุคคลธรรมดาของประเทศไทย
- การวางแผนการเงินส่วนบุคคล
- การจัดการหนี้สิน
- การลงทุนพื้นฐาน (หุ้น, กองทุนรวม, พันธบัตร)

กรุณาให้คำแนะนำที่:
1. ถูกต้องตามกฎหมายไทย
2. เข้าใจง่ายสำหรับคนทั่วไป
3. ปฏิบัติได้จริง
4. ระบุแหล่งข้อมูลอ้างอิงถ้าเป็นไปได้

หากคำถามเกี่ยวข้องกับการตัดสินใจทางการเงินที่สำคัญ ให้แนะนำให้ปรึกษาผู้เชี่ยวชาญเพิ่มเติม"""


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class FinancialAnalysisRequest(BaseModel):
    income: float
    expenses: float
    debts: List[dict]
    assets: List[dict]


class TaxDeductionRequest(BaseModel):
    annual_income: float
    marital_status: str
    dependents: int
    insurance_premium: Optional[float] = 0
    retirement_fund: Optional[float] = 0


# ============= Direct HuggingFace Implementation =============
class DirectLLMInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load model on first request"""
        if self.model is None:
            logger.info(f"Loading model {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            logger.info("Model loaded successfully!")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 2048):
        """Generate response using the model"""
        self.load_model()
        
        # Format prompt for LLaMA 2 Chat
        formatted_prompt = f"<s>[INST] <<SYS>>\n{FINANCE_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
        
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            return_full_text=False
        )
        
        return outputs[0]['generated_text']


# Initialize inference engine
direct_llm = DirectLLMInference()


# ============= TGI Client =============
async def call_tgi(prompt: str, temperature: float, max_new_tokens: int):
    """Call Text-Generation-Inference API"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{TGI_BASE_URL}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TGI API error: {response.text}"
            )
        
        result = response.json()
        return result["generated_text"]


# ============= vLLM Client (OpenAI-compatible) =============
async def call_vllm(messages: List[dict], temperature: float, max_tokens: int):
    """Call vLLM API (OpenAI-compatible)"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"vLLM API error: {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


# ============= Main Chat Endpoint =============
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for financial advice"""
    try:
        # Build conversation context
        messages = [{"role": "system", "content": FINANCE_SYSTEM_PROMPT}]
        
        if request.conversation_history:
            messages.extend(request.conversation_history)
        
        messages.append({"role": "user", "content": request.message})
        
        # Choose inference method
        if USE_VLLM:
            logger.info("Using vLLM inference")
            response_text = await call_vllm(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        elif USE_TGI:
            logger.info("Using TGI inference")
            # Format prompt for TGI
            prompt = f"<s>[INST] <<SYS>>\n{FINANCE_SYSTEM_PROMPT}\n<</SYS>>\n\n{request.message} [/INST]"
            response_text = await call_tgi(
                prompt=prompt,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens
            )
        
        else:
            logger.info("Using direct HuggingFace inference")
            response_text = direct_llm.generate(
                prompt=request.message,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens
            )
        
        return ChatResponse(
            response=response_text,
            model=MODEL_NAME
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timeout")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/financial-health")
async def analyze_financial_health(request: FinancialAnalysisRequest):
    """Analyze financial health and provide recommendations"""
    analysis_prompt = f"""วิเคราะห์สุขภาพทางการเงินจากข้อมูลต่อไปนี้:

รายได้ต่อเดือน: {request.income:,.2f} บาท
รายจ่ายต่อเดือน: {request.expenses:,.2f} บาท
หนี้สิน: {len(request.debts)} รายการ
สินทรัพย์: {len(request.assets)} รายการ

กรุณาให้:
1. คะแนนสุขภาพทางการเงิน (1-100)
2. จุดแข็ง 2-3 ข้อ
3. จุดที่ควรปรับปรุง 2-3 ข้อ
4. คำแนะนำเฉพาะเจาะจง 3-5 ข้อ

ตอบเป็นภาษาไทยในรูปแบบที่เข้าใจง่าย"""

    chat_req = ChatRequest(
        message=analysis_prompt,
        temperature=0.5,
        max_tokens=1500
    )
    
    return await chat(chat_req)


@app.post("/api/tax-deductions")
async def calculate_tax_deductions(request: TaxDeductionRequest):
    """Provide tax deduction recommendations based on Thai tax law"""
    prompt = f"""ช่วยวิเคราะห์สิทธิลดหย่อนภาษีสำหรับบุคคลในประเทศไทยที่มีข้อมูลดังนี้:

รายได้ต่อปี: {request.annual_income:,.2f} บาท
สถานภาพ: {request.marital_status}
จำนวนผู้อยู่ในอุปการะ: {request.dependents} คน
เบี้ยประกันชีวิต: {request.insurance_premium:,.2f} บาท
กองทุนสำรองเลี้ยงชีพ: {request.retirement_fund:,.2f} บาท

กรุณาให้:
1. รายการค่าลดหย่อนที่สามารถใช้ได้ทั้งหมด
2. วิธีการใช้สิทธิแต่ละรายการ
3. เอกสารที่ต้องเตรียม
4. ประมาณการภาษีที่ต้องเสีย
5. คำแนะนำเพิ่มเติมสำหรับปีถัดไป

ตอบตามกฎหมายภาษีไทยปัจจุบัน"""

    chat_req = ChatRequest(
        message=prompt,
        temperature=0.3,
        max_tokens=2000
    )
    
    return await chat(chat_req)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model": MODEL_NAME,
        "inference_mode": "vLLM" if USE_VLLM else "TGI" if USE_TGI else "Direct HF",
        "device": direct_llm.device if not (USE_VLLM or USE_TGI) else "remote"
    }
    
    return status


@app.get("/")
async def root():
    return {
        "service": "Fanato Finance LLM API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "inference": "vLLM" if USE_VLLM else "TGI" if USE_TGI else "Direct"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)