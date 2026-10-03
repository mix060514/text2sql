# 本次驗證結果

兩版都已可在這台主機執行。原工作區保留 27B 版，強模型版在相鄰
`text2sql-strong` worktree。兩版資料庫未修改，皆與 eval_manifest 的 SHA-256 相符。

| 驗證 | 強模型版 | 27B 版 |
|---|---:|---:|
| 最終離線測試 | 32/32 | 33/33 |
| 同一組 8 個代表案例 | 8/8 | 8/8 |
| 完整資料表變更後，再測月份趨勢與空結果 | 2/2 | 2/2 |
| eval_e2e 介面整合（單一 EMEA 案例） | 1/1 | 1/1 |
| 分析模型 | gpt-6-sol，medium | qwen3.8-27b-iq3-xxs |
| 格式 / 資料版本 | Codex CLI 0.157.1、ADK 1.14.1、SQLite seed-43、eval_set_v3 | ADK 1.14.1、相同 DB / dataset |

8 個案例包含地區別名、銷售量排名、訂單數排名、年份比較、
單一年份季度標籤、跨年月份趨勢、區域佔比及真正的空結果。
直接比對 SQLite oracle 與實際圖表，不用 LLM judge 判斷這些 pass/fail。

離線測試包含實際 SQLite 權限拒絕、語法錯誤、CTE/window functions、
結果上限、查詢期限、錯誤後清除資料、schema 修正、重試耗盡、整體期限、
完整事件串流、跨回合圖表隔離與完整資料表。
強模型版另外驗證不做 application retry、MCP query budget、執行 receipt
與取消後停止程序。

## 實測紀錄

- [27B 完整案例](smoke-qwen27b.json)
- [強模型完整案例](smoke-strong.json)
- [27B 最終資料表案例](smoke-qwen27b-final.json)
- [強模型最終資料表案例](smoke-strong-final.json)
- [27B 初輪原始紀錄](smoke-qwen27b-initial.json)
- [27B eval_e2e 整合紀錄](eval-integration-qwen27b.json)
- [強模型 eval_e2e 整合紀錄](eval-integration-strong.json)

eval_e2e 的單案執行確認分析、獨立 judge 與結果輸出串接正常。
這個 judge 評分不作為整體模型準確率的依據。

完整 8 案例的一次觀測總時間：強模型 152.58 秒、27B 326.58 秒。
這不是受控的速度 benchmark；不同模型、硬體、服務負載與網路影響時間。
兩個最終資料表案例另外記錄，沒有混進上述時間。

8 案例這一輪：強模型每題使用一次 sales 查詢；
27B 每題使用一次 SQL planner 和一次 report writer，沒有觸發 retry。
Codex 內部模型呼叫次數未量測，因此不能把「一次 Harness invocation」
或「一次查詢」當成「一次模型呼叫」。

## 找到與修正的問題

初輪記錄顯示 5/8，但不能把三個紅燈都當成模型失敗：

1. Canada 問「客戶是誰」，模型回傳正確名字，沒有多選訂單數。
   原 oracle 多要求一個不必要的欄位；修正比較規則，仍檢查 distinct orders 指標。
2. 月份趨勢除了正確營收，也回傳了真實訂單數與銷售量。
   改為驗證被畫出的維度/指標與順序，避免把額外正確欄位誤判為錯誤。
3. 年份總計比較被模型選成 line，是這次真正觀察到的決策錯誤。
   明確定義「年份總計比較用 bar、明確時間趨勢才用 line」，再次實測通過。

最後將完整資料表交給 deterministic result_table 產生。模型只寫解釋，
不再靠 prompt 保證每一筆資料都被重新抄寫。變更後再測 24 個月份與空結果，
兩版皆通過；整個 8 案例集的紀錄保留在資料表變更前的版本。

本機 config 的 gpt-6.1-sol 被 CLI 的 ChatGPT 登入拒絕。
gpt-6-sol 已實際測通，所以強模型版明確預設為它，沒有靜默模型 fallback。
模型可用性是帳號/客戶端的設定問題，不是 SQL 推理問題。

## 範圍與限制

這些案例證明兩條流程能執行、資料與圖表符合代表案例的 oracle，
不能證明所有自然語言敘述正確，也不能證明 27B 的整體推理能力等同強模型。
硬 gate 保證執行證據、唯讀限制、資料完整性與 schema；
指標理解或不當 SQL 分組等語意問題仍需評估。

沒有重跑全部 132 個案例，沒有部署或 push。
原本 Docker image 未包含 Codex CLI/登入；強模型版目前驗證的是主機執行方式。
Session 仍使用 InMemoryRunner，沒有加入永久 checkpoint 或 scheduler。
