from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


MAX_RETRIES = 3
ENABLE_SELF_CORRECTION = True
LLAMA_CPP_MODEL = "openai/qwen3-4b-instruct-2507"
LLAMA_CPP_API_KEY = "aaa"
LLAMA_CPP_API_BASE = "http://localhost:8081"


class JudgeResult(BaseModel):
    """評估 Agent 回答品質的結果結構"""

    reasoning: str = Field(
        ...,
        description="評分理由的簡短說明（一到兩句話），解釋為何給出此分數",
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="信心分數 (0-10)，表示回答的正確性和品質",
    )
    is_correct: bool = Field(
        ...,
        description="回答是否正確，confidence_score >= 7 視為正確",
    )


JUDGE_OUTPUT_KEY = "judge_result"  # 用於從 session state 取得結果

judge_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="judge_agent",
    description="專業的商業資料分析評估員，評估 Agent 回答的正確性與品質",
    output_key=JUDGE_OUTPUT_KEY,  # structured output 會存到 session state
    instruction="""你是一位專業的商業資料分析評估員。你的任務是評估一個 Text-to-SQL Agent 的回答品質。

## 評估維度

請從以下三個維度進行評估：

1. **答案正確性**：Agent 的回答是否與預期答案在語意上一致
2. **數據準確性**：數據、數字是否正確無誤（包括單位、格式）
3. **回答完整性**：是否完整回答了問題的所有部分

## 評分標準 (0-10分)

| 分數範圍 | 說明 |
|---------|------|
| **10分** | 完全正確，數據精確匹配，回答清晰完整 |
| **8-9分** | 基本正確，可能有微小的格式或表達差異，但不影響理解 |
| **6-7分** | 大致正確，但有部分數據偏差或遺漏次要資訊 |
| **4-5分** | 部分正確，有明顯錯誤或遺漏重要資訊 |
| **2-3分** | 大部分錯誤，只有少部分資訊正確 |
| **0-1分** | 完全錯誤、無法回答、或回答與問題無關 |

## 評估要點

- **數值比較**：注意數值的精確度，小數點差異是否在可接受範圍內
- **語意等價**：「增加 20%」與「成長 20%」語意相同，應視為正確
- **單位一致**：確認數值的單位是否一致（如百萬 vs 千）
- **SQL 結果對照**：如有提供 SQL 查詢結果，用於輔助判斷數據來源正確性

## 輸入格式

你會收到以下資訊進行評估：
- 使用者問題
- 預期正確答案
- Agent 實際回答
- SQL 查詢語句（可選）
- SQL 執行結果（可選）

## 輸出要求

- `confidence_score`: 0-10 的整數
- `is_correct`: 分數 >= 7 時為 true，否則為 false
- `reasoning`: 一到兩句話簡短說明評分理由
""",
    output_schema=JudgeResult,
)
