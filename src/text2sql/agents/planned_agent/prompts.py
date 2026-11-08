"""Domain facts are shared; control flow is implemented by each branch."""

DOMAIN = """
資料庫是 SQLite，只有 sales_data：
- "Order ID" TEXT：訂單數 = COUNT(DISTINCT "Order ID")。
- "Order Date" TEXT：YYYY-MM-DD。
- "Region" TEXT：North America、EMEA、APAC、LATAM。
- "Country" TEXT、"Customer Name" TEXT。
- "Product Category" TEXT、"Product Name" TEXT。
- "ASP" REAL：單價。
- "Quantity" INTEGER：銷售量 = SUM("Quantity")，與訂單數不同。
- "Total Revenue" REAL：銷售額、營收 = SUM("Total Revenue")。
有空格的欄位名稱必須用雙引號引用。

地區別名：北美/NA/美洲=North America；歐洲/歐非中東/中東歐非=EMEA；
亞太/Asia Pacific=APAC；南美/南美洲/拉美/中南美=LATAM。
國家別名：中國/大陸=China；德國=Germany；美國/US/USA=United States；
加拿大=Canada；印度=India；澳洲=Australia；日本=Japan；英國/UK=United Kingdom；
法國=France；巴西=Brazil；墨西哥=Mexico。

查詢原則：
- 分組彙總時 GROUP BY 對應維度，排名依問題的指標排序。
- Top N 的同分結果以名稱 ASC 作第二排序；Top 1 是穩定排序的第一筆。
- 時間趨勢按時間升冪，不可按金額排序。
- 季度：1–3 月 Q1、4–6 月 Q2、7–9 月 Q3、10–12 月 Q4。
  季度編號 = (CAST(substr("Order Date", 6, 2) AS INTEGER) + 2) / 3；不可把月份直接當季度。
- 跨年月份用 YYYY-MM；跨年季度用 YYYY-Q1..Q4 的單一時間欄位。
- 限定單一年份的季度時，標籤用 Q1..Q4，依季度排序。
- 年增率用 (今年-去年)*100.0/NULLIF(去年,0)，結果是百分比。
"""

REPORT_RULES = """
使用繁體中文回答，產品與公司名稱保留原文。
answer 要依實際 columns 與 rows 提供簡短解釋、重點或直接回答，不可編造資料或欄位。
完整資料表由程式產生；不要在 answer 重寫所有列或輸出 Markdown 表格。
空結果表示沒有符合條件的資料；不能編造答案。
只寫能直接從 rows 逐筆核對的事實，例如最高、最低、首尾數值與差額。
「普遍」「都」「均」「每個」等概括，或跨年同期比較，必須對每一個被比較的項目都成立；
有任何例外就不要概括，改寫具體哪些上升、哪些下降，或不寫這類結論。
數字加千分位；金額保留兩位小數，可另外補充 K/M 縮寫。
不要將原始 JSON 或 Plotly JSON 塞進 answer；程式負責產生圖表。

圖表規格：
- 單一結果、空結果、純文字查詢：chart_type="none"，圖表欄位用空字串。
- 多列且有數值度量：提供圖表；比較/排名用 bar、時間趨勢用 line、佔比用 pie。
- 「列出各年份的總計」是比較，用 bar；即使 x 是 Year，也不是自動用 line。
  只有問題明確要求「趨勢」或逐月/逐季時間變化時才用 line。
- x_column、y_column、color_column 必須是結果中真實欄位名稱。
- y_column 必須是數值；color_column 用於多組系列，pie 不填 color_column。
"""
