import os
import gc
import re
import json
import logging
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# 设置环境变量以优化 PyTorch 显存分配 [1]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import litellm

# --- 依赖库 ---
from modelscope import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import CrossEncoder

# ===================== 配置中心 =====================
# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] NekoBrain: %(message)s")
logger = logging.getLogger("NekoBrain")

class Config:
    # 【核心配置】本地视觉模型 (仅用于幕后OCR，不直接对话)
    LOCAL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    # 聚合API配置 (建议改为环境变量)
    AGGREGATOR_API_KEY = "sk-DuctN11czck6s758299ZoeipAjKmlhXcfhGchCZwQttQqI1o"
    AGGREGATOR_BASE_URL = "http://192.168.50.165:3000/v1"

    # 模型映射表：将路由标签映射到实际后端模型 [1]
    MODEL_MAP = {
        "general_text": "gemini-3-flash-preview", 
        "logic_king": "gemini-3-pro-preview",
        "deepthink": "gemini-3-pro-deepthink",
        "vibes_master": "MiniMaxAI/MiniMax-M2", 
        "searching": "gemini-3-flash-preview",
        "gpt-5.1": "gpt-5.1" 
    }

    # 语义路由描述
    ROUTING_DESCRIPTIONS = {
        "general_text": "General conversation, simple greetings, short questions, long essays, summarization, translation, general knowledge.",
        "logic_king": "Programming code, json, debugging, python, algorithms, variable definitions.",
        "deepthink": "Math proofs, complex physics, latex formulas, calculus, step-by-step reasoning.",
        "vibes_master": "Creative writing, roleplay, emotional support, poetry.",
        "searching": "News, current events, real-time weather, fact check."
    }

# ===================== 路由大脑 (NekoBrain) =====================
class NekoBrain:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.full_labels = list(Config.ROUTING_DESCRIPTIONS.keys())
        
        logger.info(f"📸 Initializing NekoBrain with VLM: {Config.LOCAL_MODEL_ID}...")
        self._init_local_models()
        self._init_router_model()

        # 生成参数配置
        self.generation_configs = {
            "vision": {
                "do_sample": True, "top_p": 0.8, "top_k": 20, "temperature": 0.7,
                "repetition_penalty": 1.0, "max_new_tokens": 1024,
            }
        }

    def _init_local_models(self):
        """初始化本地视觉模型 (Qwen)"""
        try:
            model_dir = snapshot_download(Config.LOCAL_MODEL_ID)
        except Exception:
            model_dir = Config.LOCAL_MODEL_ID 

        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True 
        )
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        
        # 限制分辨率以防显存溢出 [1]
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.min_pixels = 256 * 256
            self.processor.image_processor.max_pixels = 1024 * 1024

    def _init_router_model(self):
        """初始化语义路由模型"""
        self.ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

    def inject_assistant_prompt(self, messages: List[Dict]) -> List[Dict]:
        """注入系统级提示，规范 LaTeX 格式和段落"""
        new_msgs = [m.copy() for m in messages]
        injection = {
            "role": "assistant",
            "content": "好的，我会严格执行格式要求：数学公式前后加空格并使用 LaTeX，保持段落清晰。以下是我的回答：\n"
        }
        new_msgs.append(injection)
        return new_msgs

    @torch.no_grad()
    def _local_vlm_inference(self, messages: List[Dict], prompt_text: str, mode: str = "vision") -> str:
        """执行本地视觉模型推理"""
        try:
            qwen_messages = []
            for m in messages:
                if m["role"] == "system": continue 
                
                # 格式清洗
                new_m = m.copy()
                if isinstance(new_m.get("content"), list):
                    clean_content = []
                    for item in new_m["content"]:
                        if item.get("type") == "image_url":
                            img_obj = item.get("image_url")
                            url_str = img_obj.get("url") if isinstance(img_obj, dict) else str(img_obj)
                            clean_content.append({"type": "image", "image": url_str})
                        elif item.get("type") == "image":
                            clean_content.append(item)
                        else:
                            clean_content.append(item)
                    new_m["content"] = clean_content
                qwen_messages.append(new_m)
            
            qwen_messages.append({"role": "user", "content": prompt_text})

            text_input = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(qwen_messages)
            
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.vlm_model.device)

            gen_config = self.generation_configs.get(mode, self.generation_configs["vision"])
            
            generated_ids = self.vlm_model.generate(**inputs, **gen_config)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except torch.cuda.OutOfMemoryError:
            logger.error("🧱 CUDA OOM during inference! Clearing cache...")
            torch.cuda.empty_cache()
            return "GENERAL_SCENE" 
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ""
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _get_fused_decision(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        核心决策逻辑：
        1. 视觉检测 -> 本地OCR
        2. 文本提取 -> 关键词匹配或语义打分
        """
        has_image = False
        for m in messages[-2:]:
            if isinstance(m.get("content"), list):
                for item in m["content"]:
                    if item.get("type") in ["image", "image_url"]:
                        has_image = True
                        break

        extracted_text = ""
        modified_messages = messages 

        if has_image:
            logger.info("📸 [Vision Detected] Running local Qwen2.5-VL-3B analysis...")
            
            instruction = (
                "Analyze this image. "
                "If it contains document text, code, math formulas, tables, or error logs, "
                "transcribe all the content exactly (OCR). "
                "If it is a general scenery, photo of a person, or artistic image, output 'GENERAL_SCENE'."
            )
            
            vlm_output = self._local_vlm_inference(messages, instruction, mode="vision")
            
            # 视觉内容回退逻辑 [1]
            if "GENERAL_SCENE" in vlm_output or len(vlm_output.strip()) < 5:
                logger.info("🔍 [Scene/Fallback] Routing to GPT-5.1.")
                return "gpt-5.1", messages
            else:
                extracted_text = vlm_output
                clean_log = extracted_text.replace('\n', ' ')[:150]
                logger.info(f"📜 [OCR Success] Content: {clean_log}...")
                
                modified_messages = []
                for m in messages:
                    new_m = m.copy()
                    if isinstance(new_m["content"], list):
                        new_content = f"【User Uploaded Image Content (Local OCR)】\n{extracted_text}"
                        new_m["content"] = new_content
                    modified_messages.append(new_m)

        # 提取用于路由的文本
        target_text = extracted_text if extracted_text else ""
        if not target_text:
            last_msg = modified_messages[-1]
            if isinstance(last_msg["content"], str):
                target_text = last_msg["content"]
            elif isinstance(last_msg["content"], list):
                for item in last_msg["content"]:
                    if item.get("type") == "text": target_text += item.get("text", "")

        # OpenWebUI 后台任务检测：避免对自动生成的任务进行路由 [1]
        if any(re.search(p, target_text, re.I) for p in [r"### Task", r"Suggest", r"Generate a concise"]):
            return "vibes_master", modified_messages

        # 强制路由逻辑
        if has_image and extracted_text:
            if any(x in extracted_text for x in ["∫", "∑", "√", "matrix", "\\frac", "theorem", "proof"]):
                logger.info("📐 [Force Route] Math detected -> deepthink")
                return "deepthink", modified_messages
            if any(x in extracted_text for x in ["def ", "class ", "import ", "console.log", "return ", "void "]):
                logger.info("💻 [Force Route] Code detected -> logic_king")
                return "logic_king", modified_messages

        # CrossEncoder 语义打分
        ce_scores_raw = self.ce_model.predict([[target_text, v] for v in Config.ROUTING_DESCRIPTIONS.values()])
        ce_scores = {l: float(s) for l, s in zip(self.full_labels, ce_scores_raw)}
        
        sorted_scores = dict(sorted(ce_scores.items(), key=lambda item: item[1], reverse=True))
        logger.info(f"📊 [Routing Scores] {json.dumps(sorted_scores, ensure_ascii=False)}")
        
        res = max(ce_scores, key=ce_scores.get)
        logger.info(f"🚦 Route Decision: {res}")
        return res, modified_messages

    async def route(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        return await asyncio.get_event_loop().run_in_executor(self.executor, self._get_fused_decision, messages)

# ===================== FastAPI 服务 =====================
brain: Optional[NekoBrain] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain
    brain = NekoBrain()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str 
    stream: Optional[bool] = True

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    try:
        label, processed_msgs = await brain.route(req.messages)
        
        # 安全性回退：如果路由结果包含图片但模型不支持（非gpt-5.1），强制回退到 gpt-5.1 [1]
        has_image_in_processed = any(
            isinstance(m.get("content"), list) and any(i.get("type") in ["image", "image_url"] for i in m["content"])
            for m in processed_msgs
        )
        if has_image_in_processed and label != "gpt-5.1":
            logger.warning(f"⚠️ Safety Fallback: Image found in {label} route. Redirecting to gpt-5.1.")
            label = "gpt-5.1"
            processed_msgs = req.messages

        target_model = Config.MODEL_MAP.get(label, "gemini-2.5-pro") if req.model in ["auto-router-1", "auto-router-2"] else req.model
        
        extra_kwargs = {}
        if target_model == "gpt-5.1":
            extra_kwargs["reasoning_effort"] = "high" 
            logger.info("🧠 [GPT-5.1] Enforcing Reasoning Effort: HIGH")

        # 检测后台任务
        msg_str_check = str(processed_msgs)
        is_background_task = any(p in msg_str_check for p in ["### Task", "Suggest", "Generate a concise"])
        
        # 注入格式提示 (仅针对逻辑类模型)
        if not is_background_task and label in ["logic_king", "deepthink"] and target_model != "gpt-5.1":
            processed_msgs = brain.inject_assistant_prompt(processed_msgs)

        logger.info(f"🚀 Forwarding to: {target_model}")

        resp = await litellm.acompletion(
            model=target_model,
            messages=processed_msgs,
            stream=req.stream,
            api_base=Config.AGGREGATOR_BASE_URL,
            api_key=Config.AGGREGATOR_API_KEY,
            custom_llm_provider="openai",
            **extra_kwargs
        )

        if req.stream:
            async def gen():
                # --- 伪装逻辑 ---
                display_model = target_model
                if display_model == "gpt-4o":
                    display_model = "gpt-5.1"

                # 非后台任务显示前缀
                if not is_background_task:
                    prefix = f"> 😼 **NekoBrain**\n> Target: `{display_model}`\n> Label: `{label}`\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': prefix}, 'index': 0}], 'model': display_model})}\n\n"
                
                async for chunk in resp: 
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return resp
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    torch.cuda.empty_cache()
    uvicorn.run(app, host="0.0.0.0", port=2000)
