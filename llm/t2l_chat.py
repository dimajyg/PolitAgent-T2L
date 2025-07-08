from __future__ import annotations

# --- stdlib ---------------------------------------------------------------
import sys
import os
import asyncio
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional, Union, Type

# --- third‑party ----------------------------------------------------------
import torch
import json
from pydantic import BaseModel, Field, PrivateAttr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

# LangChain
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import AIMessage

# --- project imports ------------------------------------------------------
sys.path.append('/home/alisa/Documents/FakeLizzyK/PolitAgent-T2L/text-to-lora')
sys.path.append('/home/alisa/Documents/FakeLizzyK/PolitAgent-T2L/text-to-lora/src')

from hyper_llm_modulator.utils import get_layers
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils.eval_hypermod import gen_and_save_lora

from .models import register_model, get_model

logger = logging.getLogger(__name__)

# =============================================================================
# Utilities
# =============================================================================

def apply_chat_template(prompt: List[Dict[str, str]], tokenizer):
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def initialize_model(model_dir: str, peft_config: PeftConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def _load(dtype, device_map):
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if torch.cuda.is_available():
        try:
            model = _load(torch.bfloat16, "auto")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM during model loading, fallback to CPU")
                torch.cuda.empty_cache()
                model = _load(torch.float32, "cpu")
            else:
                raise
    else:
        model = _load(torch.float32, "cpu")

    return PeftModel(model, peft_config), tokenizer

# =============================================================================
# Tool base class and concrete tools
# =============================================================================

class T2LAdapterInput(BaseModel):
    prompt: str = Field(description="The input prompt for the T2L adapter")


class _BaseT2LTool(BaseTool):
    args_schema: Type[BaseModel] = T2LAdapterInput

    # приватные (не валидируются, не требуются в __init__)
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _lora_gen_fn: Any = PrivateAttr()
    _task_prompt: str = PrivateAttr(default="")
    _adapter_name: str = ""

    def __init__(self,
                 *,
                 lora_gen_fn,
                 task_prompt,
                 model,
                 tokenizer,
                 **data):
        super().__init__(**data)           # в BaseTool отправляем только ‘официальные’ поля
        self._lora_gen_fn = lora_gen_fn
        self._task_prompt = task_prompt
        self._model = model
        self._tokenizer = tokenizer


    def _run(self, prompt: str) -> str:  # noqa: D401
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            lora_dir = f"/tmp/gen_lora/{self._adapter_name}"
            self._lora_gen_fn(lora_dir=lora_dir, task_desc=f"{self._task_prompt}\\n{prompt}")
            self._model.load_adapter(lora_dir, self._adapter_name)
            self._model.set_adapter(self._adapter_name)

            chat_prompt = apply_chat_template([{"role": "user", "content": prompt}], self._tokenizer)
            inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            return self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class T2LActTool(_BaseT2LTool):
    name: str = "t2l_act_adapter"
    description: str = (
        "Use this tool when you need to choose diplomatic action, order, or strategic move. "
        "Best for task-choosing, decision-making and strategic action decision."
    )
    _adapter_name: str = "act"


class T2LTalkTool(_BaseT2LTool):
    name: str = "t2l_talk_adapter"
    description: str = (
        "Use this tool when you need to generate diplomatic messages, "
        "negotiations, or conversations. Best for communication and dialogue tasks."
    )
    _adapter_name: str = "talk"


class T2LSelfQuestionTool(_BaseT2LTool):
    name: str = "t2l_self_question_adapter"
    description: str = (
        "Use this tool to pose incisive self‑queries, surface hidden assumptions and generate "
        "fresh insights that will strengthen later decisions."
    )
    _adapter_name: str = "selfq"

# =============================================================================
# Main wrapper
# =============================================================================

@register_model("t2l")
class T2LChatModel:
    """LangChain‑compatible wrapper for Text‑to‑LoRA with robust structured output."""

    def __init__(self, checkpoint_path: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 256, llm_model: str = "mistral", **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_model_type = llm_model
        self.checkpoint_path = checkpoint_path or "/home/alisa/Documents/FakeLizzyK/PolitAgent-T2L/text-to-lora/trained_t2l/gemma_2b_t2l"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._initialize_t2l()
        self._initialize_llm_model(**kwargs)
        self.agent = None

        self.task_descriptions = {
            "act": (
                "You are a strategic decision-maker in diplomatic negotiations. "
                "Analyze the current situation, consider all available options, "
                "and choose the most effective actions that align with your goals, "
                "while managing relationships and risks."
                ),
            "talk": (
                "You are a skilled diplomat engaging in negotiations and conversations. "
                "Craft persuasive, relationship-aware messages that advance your strategic objectives "
                "while maintaining appropriate diplomatic tone and building trust or applying pressure as needed."
                ),
            "self_question": (
                "You are a strategic analyst conducting internal reflection and planning. "
                "Question your assumptions, evaluate your current position, assess risks and opportunities, "
                "and develop deeper insights about the situation and your strategic options. "
                "Ensure your questions are informed, specific, and actionable. "
            ),
        }

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_t2l(self):
        with open(f"{self.checkpoint_path}/adapter_config.json", "r") as f:
            adapter_cfg = json.load(f)

        (
            self.args,
            self.hypermod,
            self.t2l_model,
            self.t2l_tokenizer,
            self.emb_model,
            self.emb_tokenizer,
            self.task_desc_format_fn,
            self.pooling_fn,
        ) = load_hypermod_checkpoint(f"{self.checkpoint_path}/hypermod.pt", self.device)

        self.peft_config = PeftConfig.from_pretrained(self.checkpoint_path)
        layer_idxs = torch.tensor(range(len(get_layers(self.t2l_model))), dtype=torch.long, device=self.device)

        self._gen_and_save_lora = partial(
            gen_and_save_lora,
            model_dir=adapter_cfg["base_model_name_or_path"],
            device=self.device,
            layer_indices=layer_idxs,
            emb_model=self.emb_model,
            emb_tokenizer=self.emb_tokenizer,
            task_desc_format_fn=self.task_desc_format_fn,
            pooling_fn=self.pooling_fn,
            hypermod=self.hypermod,
        )

        self.model, self.tokenizer = self.t2l_model, self.t2l_tokenizer

    def _initialize_llm_model(self, **kwargs):
        self.llm = get_model(model_name=self.llm_model_type, temperature=self.temperature, max_tokens=self.max_tokens, **kwargs)

    # ------------------------------------------------------------------
    # Agent helpers
    # ------------------------------------------------------------------

    def _create_tools(self):
        return [
            T2LActTool(
                lora_gen_fn=self._gen_and_save_lora,
                task_prompt=self.task_descriptions["act"],
                model=self.model,
                tokenizer=self.tokenizer
                ),
            T2LTalkTool(
                lora_gen_fn=self._gen_and_save_lora,
                task_prompt=self.task_descriptions["talk"],
                model=self.model,
                tokenizer=self.tokenizer
                ),
            T2LSelfQuestionTool(
                lora_gen_fn=self._gen_and_save_lora,
                task_prompt=self.task_descriptions["self_question"],
                model=self.model,
                tokenizer=self.tokenizer
                ),
        ]

    def _create_agent(self):
        # --- custom parsing-error handler ---------------------------------
        def _fix_parsing_err(err):
            """Clean LLM output so default parser can succeed."""
            raw = getattr(err, "llm_output", str(err))
            # убираем строки от LangChain
            cleaned = re.sub(r"For troubleshooting.*", "", raw)

            # Если модель выдала и Action, и Final Answer, оставляем Action-блок
            if "Action:" in cleaned and "Final Answer:" in cleaned:
                # обрезаем всё после Final Answer (по умолчанию ReAct берёт Action)
                cleaned = cleaned.split("Final Answer:")[0].rstrip()
            return cleaned
        if self.agent is None:
            self.agent = initialize_agent(
                tools=self._create_tools(),
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                # verbose=True,
                handle_parsing_errors=_fix_parsing_err,
                max_iterations=15,
            )

    # ------------------------------------------------------------------
    # Public invoke
    # ------------------------------------------------------------------

    def invoke(self, input_data, **kwargs):
        self._create_agent()
        if isinstance(input_data, list):
            prompt = "\n".join([m["content"] for m in input_data if "content" in m])
        else:
            prompt = str(input_data)
        answer = self.agent.invoke(prompt).get('output', " ")

        print(f"!!!!!!!!!!!!!! {answer} !!!!!!!!!!!!!!!")

        return AIMessage(content=answer)

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def with_structured_output(self, schema: Type[BaseModel], *, max_retries: int = 1):
            base_parser, fix_parser = PydanticOutputParser(pydantic_object=schema), None  # fix_parser init later
            fmt_instr = base_parser.get_format_instructions()
            parent = self

            class _StructuredInvoker:
                def invoke(self_inner, inp):
                    # Ensure agent exists
                    parent._create_agent()
                    raw = inp if isinstance(inp, str) else (" ".join(m.get("content", "") for m in inp) if isinstance(inp, list) else (" ".join(m.content for m in inp.messages) if hasattr(inp, "messages") else str(inp)))
                    prompt = f"{raw}\n\n{fmt_instr}"
                    answer = parent.agent.run(prompt)
                    txt = answer.strip()
                    m = re.search(r"```json\s*(.*?)\s*```", txt, re.S)
                    if m:
                        txt = m.group(1)
                    try:
                        return base_parser.parse(txt)
                    except Exception as err:
                        nonlocal fix_parser
                        if fix_parser is None:
                            fix_parser = OutputFixingParser.from_llm(parser=base_parser, llm=parent.llm)
                        last_err = err
                        for _ in range(max_retries):
                            try:
                                fixed = fix_parser.parse(txt)
                                return base_parser.parse(fixed)
                            except Exception as err2:
                                last_err = err2
                        raise last_err
            return _StructuredInvoker()
