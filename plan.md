# Implementation Plan - Text to SQL Agents

## 當前狀態分析 (Eval 5 Results)

**成功率**: 62.2% (51/82 正確)
**問題數量**: 31 個失敗案例

### 錯誤分類
1. **Dataframe 格式問題** (15 cases): Ground truth 使用 `'2023-1'` 但 SQL 返回 `'2023-01'`
2. **Region 別名問題** (5 cases): "Europe, Middle East, and Africa" → 應該映射到 `'EMEA'`
3. **Ground Truth 錯誤** (5 cases): India/Germany 的客戶資料與實際不符
4. **季度提取問題** (1 case): SQLite 的 `strftime('%q')` 返回 `null`
5. **其他問題** (5 cases): 需進一步分析

---

## Agent V6 實作進度 (LoopAgent 方案)

### 目標
使用 Google ADK 的 `LoopAgent` 實現真正的程式化 retry loop,而非僅依賴 prompt 的自我修正。

### 架構設計
```
SequentialAgent (planned_agent_v6):
  1. initial_sql_generator
     - 生成初始 SQL 查詢
     - 使用 check_sql_syntax 驗證語法
     - 寫入 state: sql_query
  
  2. refinement_loop (LoopAgent, max_iterations=3)
     a. sql_executor_in_loop
        - 讀取 state: sql_query
        - 執行 execute_sql
        - 如果結果非空: 調用 exit_retry_loop tool (escalate=True)
        - 如果結果為空: 輸出 "RETRY"
        - 寫入 state: sql_result
     
     b. sql_refiner_in_loop
        - 讀取 state: sql_query, sql_result
        - 如果 sql_result == "RETRY": 分析並修正 SQL
        - 寫入 state: sql_query (覆蓋)
  
  3. response_formatter
     - 讀取 state: sql_query, sql_result
     - 格式化為繁體中文回答
```

### 關鍵技術點
1. **State 管理**: 使用 `output_key` 在 sub-agents 間共享狀態
2. **Loop 終止**: 使用 `tool_context.actions.escalate = True` 停止 loop
3. **避免 KeyError**: 使用 SequentialAgent 確保 state 變數在被引用前已初始化
4. **include_contents='none'**: 避免 conversation history 污染

### 當前問題 ⚠️

**Blocker**: `SequentialAgent` 在第一個 sub-agent (`initial_sql_generator`) 完成後就停止,沒有繼續執行 `refinement_loop` 和 `response_formatter`。

**測試結果**:
```
=== Testing Agent V6 (Complete 3-Step Flow) ===
[1] Call: check_sql_syntax
[2] Resp: check_sql_syntax -> {'result': 'Valid'}...

=== FINAL ===
SELECT SUM("Total Revenue") AS "Sales" FROM sales_data WHERE "Region" = 'North America'
```

**分析**:
- Agent 只返回 SQL 查詢字串
- 沒有執行 SQL (沒有看到 `execute_sql` 調用)
- 沒有格式化為繁體中文

**可能原因**:
1. SequentialAgent 的行為與預期不同?
2. 需要特殊配置讓 SequentialAgent 繼續執行所有 sub-agents?
3. LoopAgent 在 SequentialAgent 中的使用有特殊要求?

**下一步選項**:
- [ ] 深入研究 ADK SequentialAgent 文檔
- [ ] 查看 ADK 原始碼了解 SequentialAgent 執行邏輯
- [ ] 考慮放棄 LoopAgent,改用其他方式實作 retry
- [ ] 先評估 Agent V5 (prompt-based retry) 的效果

---

## Agent V5 設計目標 (Fallback 方案)

### 核心改進
1. **Retry Loop 機制**: 允許 Agent 自我修正 SQL 錯誤
2. **結構化輸出**: 統一 Agent 回傳格式,簡化 Eval 解析
3. **Region/Country 別名映射**: 在 Agent instruction 中加入完整映射表
4. **更好的錯誤處理**: 檢測空結果並提示可能的別名問題

---

## 實施計劃

### Phase 1: Agent V5 架構設計

#### 1.1 建立 Retry Loop Agent
- [ ] 創建 `src/text2sql/agents/planned_agent/agent_v5.py`
- [ ] 設計 Retry Loop 工作流程:
  ```
  Loop (max_retries=3):
    1. 生成 SQL
    2. 檢查語法 (check_sql_syntax)
    3. 執行 SQL (execute_sql)
    4. 驗證結果:
       - 如果結果為空 → 分析可能原因 → 重試
       - 如果有語法錯誤 → 修正 → 重試
       - 如果結果合理 → 返回
  ```
- [ ] 在 `config.py` 中定義超參數:
  ```python
  MAX_RETRIES = 3
  ENABLE_SELF_CORRECTION = True
  ```

#### 1.2 結構化輸出格式

定義標準 JSON 輸出格式:
```json
{
  "sql_query": "SELECT ...",
  "sql_result": [...],
  "answer": "根據查詢結果...",
  "retry_count": 1,
  "error_log": []
}
```

#### 1.3 Region/Country 別名映射
在 Agent instruction 中加入:
```python
REGION_ALIASES = {
    "North America": ["North America", "北美", "北美洲", "NA"],
    "EMEA": ["EMEA", "歐洲中東非洲", "歐非中東", "Europe, Middle East, and Africa"],
    "APAC": ["APAC", "亞太區", "亞太地區", "Asia Pacific"],
    "LATAM": ["LATAM", "拉美", "南美", "拉丁美洲", "Latin America"]
}

COUNTRY_ALIASES = {
    "China": ["China", "中國", "大陸", "PRC"],
    "Germany": ["Germany", "德國", "德意志"],
    ...
}
```

**重要提示**: Agent 需要理解:
- 當用戶說 "Europe, Middle East, and Africa" 時,SQL 應使用 `Region = 'EMEA'`
- 當用戶說 "亞太區" 時,SQL 應使用 `Region = 'APAC'`

---

### Phase 2: Evaluation 改進

#### 2.1 修正 `eval5.py` 的比較邏輯
- [ ] 修改 `compare_answers_smart()` 函數:
  - **Dataframe 比較**: 支援 `'2023-1'` 與 `'2023-01'` 的寬鬆匹配
  - **提取結構化輸出**: 優先從 JSON 格式中提取 `sql_result`
  - **改進數字比較**: 處理浮點數精度問題 (如 `324390.47` vs `324390.47000000003`)

#### 2.2 創建 `eval6.py`
- [ ] 基於 `eval5.py` 創建新版本
- [ ] 支援 Agent V5 的結構化輸出
- [ ] 改進錯誤分類和報告:
  ```python
  error_categories = {
      "region_alias_error": [],
      "sql_syntax_error": [],
      "empty_result_error": [],
      "dataframe_format_mismatch": [],
      "ground_truth_error": []
  }
  ```

---

### Phase 3: Ground Truth 修正

#### 3.1 修正 `generate_ground_truth.py`
- [ ] **修正 Dataframe 格式**:
  ```python
  # 修改月份格式從 '2023-1' 改為 '2023-01'
  ans_dict = {
      f"{int(y)}-{int(m):02d}": round(v, 2)  # 加入 :02d 格式化
      for y, m, v in zip(ans["year"], ans["month"], ans["Total Revenue"])
  }
  ```

- [ ] **驗證 Ground Truth 正確性**:
  - 執行腳本重新生成 `eval_set_v3.jsonl`
  - 手動驗證 India/Germany 客戶資料
  - 修正任何不一致的資料

#### 3.2 創建資料驗證腳本
- [ ] 創建 `scripts/validate_ground_truth.py`:
  - 連接資料庫
  - 逐一驗證每個 ground truth 答案
  - 輸出不一致的案例

---

### Phase 4: 整合與測試

#### 4.1 整合測試
- [ ] 使用 Agent V5 執行 `eval6.py`
- [ ] 目標成功率: **> 85%**
- [ ] 分析剩餘錯誤並迭代改進

#### 4.2 文件更新
- [ ] 更新 `README.md` 說明 Agent V5 的改進
- [ ] 創建 `docs/agent_v5_design.md` 詳細設計文件
- [ ] 更新 MLflow 實驗記錄

---

## 技術細節

### Agent V5 Retry Loop 實作範例

```python
class SQLRetryAgent:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
    
    async def execute_with_retry(self, question):
        error_log = []
        
        for attempt in range(self.max_retries):
            # 1. 生成 SQL
            sql_query = await self.generate_sql(question, error_log)
            
            # 2. 檢查語法
            syntax_check = check_sql_syntax(sql_query)
            if not syntax_check["valid"]:
                error_log.append({
                    "attempt": attempt,
                    "error": "syntax_error",
                    "message": syntax_check["error"]
                })
                continue
            
            # 3. 執行 SQL
            result = execute_sql(sql_query)
            
            # 4. 驗證結果
            if self.is_valid_result(result, question):
                return {
                    "sql_query": sql_query,
                    "sql_result": result,
                    "retry_count": attempt,
                    "error_log": error_log
                }
            else:
                error_log.append({
                    "attempt": attempt,
                    "error": "empty_or_invalid_result",
                    "hint": self.suggest_fix(question, sql_query, result)
                })
        
        # 達到最大重試次數
        return {"error": "max_retries_exceeded", "error_log": error_log}
```

---

## 預期成果

1. **Agent V5**: 具備自我修正能力的 SQL Agent
2. **Eval6**: 更精確的評估框架
3. **Ground Truth V3**: 修正格式和資料錯誤
4. **成功率提升**: 從 62.2% → 85%+
5. **更好的可維護性**: 結構化輸出和錯誤日誌

---

## 時間估算

- Phase 1: Agent V5 設計與實作 - **2-3 小時**
- Phase 2: Evaluation 改進 - **1-2 小時**
- Phase 3: Ground Truth 修正 - **1 小時**
- Phase 4: 整合測試與迭代 - **2 小時**

**總計**: 約 6-8 小時

---

## 下一步行動

1. ✅ 分析 Eval 5 結果
2. ✅ 撰寫實施計劃
3. ⏭️ 開始實作 Agent V5
4. ⏭️ 修正 Ground Truth
5. ⏭️ 創建 Eval6 並測試
