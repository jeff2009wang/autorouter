import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import json
import traceback
import asyncio
import torch
import numpy as np
import time
import hashlib
from functools import lru_cache
from collections import OrderedDict
from torch.cuda import amp
from typing import List, Dict, Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from modelscope import snapshot_download
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from sentence_transformers import CrossEncoder

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import litellm

# ===================== 配置中心 =====================
logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NekoBrain")

if os.getenv("NEKOBRAIN_DEBUG", "false").lower() == "true":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# 抑制 litellm 的冗余输出
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True

# 1. 本地路由大脑模型
LOCAL_ROUTER_ID = "Qwen/Qwen2.5-7B-Instruct"

# 2. 在线视觉模型配置
ONLINE_VLM_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# 聚合API配置
AGGREGATOR_API_KEY = "sk-DuctN11czck6s758299ZoeipAjKmlhXcfhGchCZwQttQqI1o"
AGGREGATOR_BASE_URL = "http://192.168.50.165:3000/v1"

MODEL_MAP = {
    "flash_smart": "gemini-3-flash-preview",
    "pro_advanced": "gemini-3-pro",
    "code_technical": "gpt-5-codex-high",
    "code_architect": "claude-4-opus",
    "logic_reasoning": "gemini-3-pro-deepthink",
    "expert_xhigh": "gpt-5.2-xhigh"
}

# ===================== 路由大脑 =====================
class LRUCache:
    """简单的LRU缓存实现，用于路由结果缓存"""
    def __init__(self, max_size: int = 256):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Tuple[str, List[Dict]]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Tuple[str, List[Dict]]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()

class NekoBrain:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 优化：2060 12GB显存有限，减少并发线程数
        self.executor = ThreadPoolExecutor(max_workers=4) 
        self.enable_perf_logging = True
        
        # 添加路由结果缓存（256条，约占用10-20MB内存）
        self.route_cache = LRUCache(max_size=256)
        
        self.full_labels = list(MODEL_MAP.keys())
        
        # 快速路径关键词映射（提高准确度和速度）
        self.quick_keywords = {
            "code_technical": ["def ", "class ", "import ", "function", "sql", "query", "python", "javascript", "java", "c++", "代码", "编程", "debug"],
            "code_architect": ["architecture", "design pattern", "system design", "microservice", "架构", "设计模式"],
            "logic_reasoning": ["prove", "theorem", "calculate", "solve", "equation", "integral", "微分", "积分", "证明", "计算"],
            "pro_advanced": ["creative", "story", "poem", "creative writing", "创作", "故事", "诗歌", "analysis"],
            "flash_smart": ["hello", "hi", "thanks", "你好", "谢谢"],
            "expert_xhigh": ["research", "paper", "academic", "research", "研究", "学术"]
        }
        
        logger.info("🧠 Initializing NekoBrain (Online VLM + Local Router)...")
        logger.info(f"👁️ Using Online VLM via Aggregator: {ONLINE_VLM_ID}")

        # --- 加载本地路由模型 (Router) ---
        try:
            logger.info("🧠 Loading Local Router (Qwen2.5-7B-Instruct)...")
            router_dir = snapshot_download(LOCAL_ROUTER_ID)
            
            bnb_config_router = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # 优化显存使用：限制显存分配，优先使用CPU卸载
            max_memory = {0: "10GB"} if self.device == "cuda" else None

            self.router_model = AutoModelForCausalLM.from_pretrained(
                router_dir,
                torch_dtype=torch.float16,
                quantization_config=bnb_config_router,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # 尝试使用torch.compile加速（PyTorch 2.0+，可选）
            try:
                if hasattr(torch, 'compile') and self.device == "cuda":
                    logger.info("⚡ Using torch.compile for optimization...")
                    self.router_model = torch.compile(self.router_model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile not available or failed: {e}")
            self.router_tokenizer = AutoTokenizer.from_pretrained(router_dir, trust_remote_code=True)
            logger.info("✅ Router model loaded successfully")
            
            self._warmup_models()
                
        except Exception as e:
            logger.error(f"Failed to load Router: {e}")
            raise e

        # --- 辅助向量模型 (CPU 运行) ---
        self.ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device="cpu")
        
        logger.info("✅ NekoBrain initialization complete")

    def _warmup_models(self):
        logger.info("🔥 Warming up models...")
        try:
            dummy_text = "Hello, this is a warmup test."
            self._get_router_scores(dummy_text)
            logger.info("✅ Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def inject_assistant_prompt(self, messages: List[Dict]) -> List[Dict]:
        new_msgs = [m.copy() for m in messages]
        injection = {
            "role": "assistant",
            "content": "I will provide a professional solution. For code, I will optimize it. For math, I use LaTeX.\n"
        }
        new_msgs.append(injection)
        return new_msgs

    def _online_vlm_inference(self, messages: List[Dict], prompt_text: str) -> str:
        try:
            logger.info(f"👁️ Sending image to online VLM ({ONLINE_VLM_ID}) for OCR...")
            
            target_msg = None
            for m in reversed(messages):
                if isinstance(m.get("content"), list):
                    for item in m["content"]:
                        if item.get("type") in ["image", "image_url"]:
                            target_msg = m
                            break
                if target_msg: break
            
            if not target_msg: return ""

            vlm_messages = [
                target_msg,
                {"role": "user", "content": prompt_text}
            ]

            response = litellm.completion(
                model=f"openai/{ONLINE_VLM_ID}", 
                messages=vlm_messages,
                api_base=AGGREGATOR_BASE_URL,
                api_key=AGGREGATOR_API_KEY,
                max_tokens=1024,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            logger.info("✅ Online OCR complete.")
            return result
        except Exception as e:
            logger.error(f"Online VLM Error: {e}")
            return ""

    def _quick_keyword_match(self, text: str) -> Optional[str]:
        """快速关键词匹配，返回最可能的标签（用于加速简单场景）"""
        text_lower = text.lower()
        scores = {}
        for label, keywords in self.quick_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if matches > 0:
                scores[label] = matches
        
        if scores:
            best_label = max(scores, key=scores.get)
            # 只有匹配度足够高（>=2个关键词）才使用快速路径
            if scores[best_label] >= 2:
                return best_label
        return None
    
    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """将原始分数归一化"""
        scores = list(raw_scores.values())
        if not scores: return raw_scores
        
        min_score, max_score = min(scores), max(scores)
        
        # 避免除以零
        if max_score == min_score: 
            return {label: 5.0 for label in raw_scores.keys()}
        
        # 【关键修复】这里之前写成了 k，导致 UnboundLocalError，现已修正为 label
        return {
            label: 1.0 + 9.0 * (score - min_score) / (max_score - min_score) 
            for label, score in raw_scores.items()
        }

    def _get_embedding_scores(self, text: str) -> Dict[str, float]:
        DESCRIPTIONS = {
            "flash_smart": "General assistance, daily chat, simple questions, greetings.",
            "pro_advanced": "Complex analysis, creative writing, nuanced language understanding.",
            "code_technical": "Writing code, Python/C++/Java, SQL queries, debugging scripts.",
            "code_architect": "System design, software architecture, explaining technical concepts.",
            "logic_reasoning": "Advanced mathematics, physics, logic puzzles, scientific reasoning.",
            "expert_xhigh": "Specialized professional research, high-context analysis."
        }
        ce_scores_raw = self.ce_model.predict([[text, v] for v in DESCRIPTIONS.values()])
        raw_scores = {l: float(s) for l, s in zip(self.full_labels, ce_scores_raw)}
        return self._normalize_scores(raw_scores)

    @torch.no_grad()
    def _get_router_scores(self, text: str) -> Dict[str, float]:
        start_time = time.time()
        try:
            # 使用完整的context以确保准确度
            context_segment = text[:800]
            
            # 详细完整的prompt，确保模型充分理解每个类别的含义
            prompt = (
                "Rate the user input for EACH category below. You MUST rate ALL 6 categories.\n"
                "Score: 1 = Not relevant, 10 = Perfect match\n\n"
                "Categories:\n"
                "1. flash_smart: General chat, greetings, simple questions, daily conversation\n"
                "2. pro_advanced: Complex analysis, creative writing, nuanced language understanding, detailed explanations\n"
                "3. code_technical: Programming, debugging, SQL queries, writing code in Python/C++/Java, technical scripts\n"
                "4. code_architect: System design, software architecture, explaining technical concepts, architectural patterns\n"
                "5. logic_reasoning: Math proofs, physics problems, logic puzzles, step-by-step reasoning, calculus, theorems\n"
                "6. expert_xhigh: Professional research, academic papers, high-context analysis, specialized knowledge\n\n"
                f"User Input: \"{context_segment}\"\n\n"
                "Output ALL 6 ratings in format: label:X (one per line, where X is a number from 1 to 10)."
            )
            messages = [{"role": "system", "content": "You are a precise classifier. Rate each category from 1 to 10 based on relevance."}, {"role": "user", "content": prompt}]
            
            text_input = self.router_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.router_tokenizer([text_input], return_tensors="pt").to(self.router_model.device)
            
            # 优化：减少max_new_tokens（从120降到80），使用KV cache，优化生成速度
            with amp.autocast():
                generated_ids = self.router_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=80,  # 减少生成token数，加速推理
                    temperature=0.1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.router_tokenizer.eos_token_id,
                    use_cache=True  # 启用KV cache
                )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.router_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 不使用正则匹配，改用字符串分割和解析，提高准确率
            scores = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if ':' not in line:
                    continue
                    
                # 尝试多种格式：label:score, label: score, label=score等
                for separator in [':', '=', ' ']:
                    if separator in line:
                        parts = line.split(separator, 1)
                        if len(parts) == 2:
                            potential_label = parts[0].strip().lower()
                            potential_score = parts[1].strip()
                            
                            # 检查是否是已知标签
                            for label in self.full_labels:
                                if label.lower() in potential_label or potential_label in label.lower():
                                    # 尝试提取数字分数（不使用正则）
                                    score_str = ""
                                    for char in potential_score:
                                        if char.isdigit() or char == '.':
                                            score_str += char
                                        elif char in [' ', '\t'] and score_str:
                                            break
                                        elif char not in [' ', '\t'] and not (char.isdigit() or char == '.'):
                                            if score_str:
                                                break
                                    
                                    if score_str:
                                        try:
                                            score = float(score_str)
                                            # 确保分数在合理范围内
                                            if 0 <= score <= 10:
                                                scores[label] = score
                                                break
                                        except ValueError:
                                            continue
            
            for label in self.full_labels:
                if label not in scores: scores[label] = 1.0
            
            if self.enable_perf_logging:
                logger.info(f"⚡ Router: {(time.time() - start_time)*1000:.1f}ms")
            
            return scores
        except Exception as e:
            logger.error(f"Router scoring error: {e}")
            return {label: 1.0 for label in self.full_labels}

    def _get_text_hash(self, text: str) -> str:
        """生成文本的hash用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_fused_decision(self, messages: List[Dict]) -> tuple[str, List[Dict]]:
        decision_start = time.time()
        target_text = ""
        modified_messages = messages 
        
        has_image = any(
            isinstance(m.get("content"), list) and any(item.get("type") in ["image", "image_url"] for item in m["content"])
            for m in messages[-2:]
        )
        
        if has_image:
            logger.info("📸 Image detected. Starting Online OCR...")
            extracted_text = self._online_vlm_inference(messages, "Detailed transcription of this image.")
            target_text = extracted_text
            modified_messages = []
            for m in messages:
                new_m = m.copy()
                if isinstance(new_m.get("content"), list):
                    new_m["content"] = f"【System Note: Image Content (OCR):】\n{extracted_text}"
                modified_messages.append(new_m)
        else:
            last_msg = messages[-1]
            if isinstance(last_msg["content"], str):
                target_text = last_msg["content"]
            elif isinstance(last_msg["content"], list):
                for item in last_msg["content"]:
                    if item.get("type") == "text": target_text += item.get("text", "")
        
        # 优化：检查缓存（跳过图片场景，因为OCR结果可能不同）
        if not has_image and target_text:
            text_hash = self._get_text_hash(target_text)
            cached_result = self.route_cache.get(text_hash)
            if cached_result:
                logger.info(f"⚡ Cache hit! Route: {cached_result[0]} ({((time.time() - decision_start)*1000):.1f}ms)")
                return cached_result
        
        # 优化：快速路径 - 对简单场景使用关键词匹配
        if target_text and len(target_text) < 500:
            quick_label = self._quick_keyword_match(target_text)
            if quick_label:
                logger.info(f"⚡ Quick path: {quick_label} ({((time.time() - decision_start)*1000):.1f}ms)")
                result = (quick_label, modified_messages)
                if not has_image and target_text:
                    text_hash = self._get_text_hash(target_text)
                    self.route_cache.put(text_hash, result)
                return result

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            embedding_future = executor.submit(self._get_embedding_scores, target_text)
            router_future = executor.submit(self._get_router_scores, target_text)
            embedding_scores = embedding_future.result()
            router_scores = router_future.result()
        
        # 优化：改进评分融合算法（加权平均，router权重更高因为更准确）
        final_scores = {}
        for label in self.full_labels:
            emb_score = embedding_scores.get(label, 5.0)
            router_score = router_scores.get(label, 1.0)
            # Router权重0.6，Embedding权重0.4（可以根据效果调整）
            final_scores[label] = 0.6 * router_score + 0.4 * emb_score
        
        best_label = max(final_scores, key=final_scores.get)
        
        # 缓存结果（图片场景不缓存）
        if not has_image and target_text:
            result = (best_label, modified_messages)
            self.route_cache.put(text_hash, result)
        
        # 【关键恢复】恢复了您需要的详细表格输出逻辑
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("="*60)
            logger.debug(f"Input: {target_text[:200]}...")
            logger.debug("-"*60)
            logger.debug("Scoring Details:")
            for label in self.full_labels:
                logger.debug(
                    f"  {label:15} | Emb: {embedding_scores.get(label, 0):.2f} | "
                    f"Router: {router_scores.get(label, 0):.2f} | "
                    f"Final: {final_scores[label]:.2f}"
                )
            logger.debug("-"*60)
            logger.debug(f"Final Decision: {best_label} ({(time.time() - decision_start)*1000:.1f}ms)")
            logger.debug("="*60)
        else:
            logger.info(f"🎯 Route: {best_label} ({(time.time() - decision_start)*1000:.1f}ms)")
        
        return best_label, modified_messages

    async def route(self, messages: List[Dict]) -> tuple[str, List[Dict]]:
        return await asyncio.get_event_loop().run_in_executor(self.executor, self._get_fused_decision, messages)

# ===================== FastAPI 服务 =====================
brain: NekoBrain = None
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
        # 统一处理auto_router模型名称（auto_router1, auto_router2等都视为auto_router）
        if req.model and req.model.startswith("auto"):
            req.model = "auto_router"
        
        label, processed_msgs = await brain.route(req.messages)
        
        target_model = MODEL_MAP.get(label, "gemini-3-flash-preview")
        if "code" in label or "logic" in label:
            processed_msgs = brain.inject_assistant_prompt(processed_msgs)

        logger.info(f"🚀 Routing to: {target_model}")

        resp = await litellm.acompletion(
            model=f"openai/{target_model}", 
            messages=processed_msgs,
            stream=req.stream,
            api_base=AGGREGATOR_BASE_URL,
            api_key=AGGREGATOR_API_KEY
        )

        if req.stream:
            async def gen():
                prefix = f"> 🧠 **NekoBrain**\n> Route: `{target_model}`\n"
                yield f"data: {json.dumps({'choices': [{'delta': {'content': prefix}, 'index': 0}], 'model': target_model})}\n\n"
                async for chunk in resp: 
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return resp
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2000)
