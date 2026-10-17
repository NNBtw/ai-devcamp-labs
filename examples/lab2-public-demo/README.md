# Lab 2: anonymized session and memory demo

This example contains **5 memories, 3 sessions, and 46 ordered session events** derived from reviewed Social Spark lab records. The containing document replaces the original snapshot metadata; all 54 memory/session/event records remain represented.

## What this demonstrates

- **S001 (20 events):** an English Python debugging draft, image references, a human confirmation request, approval, and a simulated posting result.
- **S002 (20 events):** a Traditional Chinese type-hints draft for a fictional Python learning community, followed by the same confirmation flow.
- **S003 (6 events):** a session continuation check using the non-secret `LAB2-SESSION-CHECK` marker.
- **M001–M005:** synthetic preferences used in memory tests. They do not establish the author's personal preferences. The two posting sessions returned an empty memory search; this dataset does not demonstrate retrieval of these five memories into those sessions.

`DRY_RUN=true` was active. No LinkedIn post was published. Image and simulated post links use the reserved `example.invalid` domain and intentionally do not resolve. No image files are included.

## Privacy transformations

- Removed all 28 thought-signature fields, including the copies in duplicated raw events.
- Omitted duplicate `rawEvent` payloads, precise timestamps, expiration times, usage telemetry, and original deployment/snapshot metadata.
- Replaced resource names, user/thread/invocation/tool-call identifiers, synthetic canary suffixes, container paths, and output URLs with consistent demo values. Call, confirmation, and response references remain linked.
- Preserved event order, readable prompts and generated programming content, tool arguments/results, state changes, and synthetic memory facts (with their identifying suffix normalized).

The result is an illustrative derivative, **not an API response fixture or restorable backup**. It excludes memory revision history, image contents, Cloud Logging, credentials/configuration outside these records, and other cloud resources. The readable source records were manually reviewed; automated checks verify record coverage, signature/identifier removal, and confirmation/DRY_RUN relationships. These checks do not certify unrelated project data.

## Files

- `demo-data.json`: anonymized educational records.
- `manifest.json`: public file hashes and validation summary. It contains no mapping back to private identifiers.

## 中文說明

本資料包含匿名化後的 5 筆 memories、3 個 sessions 與 46 筆依序排列的 events。記憶中的偏好是模擬測試資料；兩個發文流程來自實際課堂操作，但全程使用 DRY_RUN，沒有真正發布至 LinkedIn。兩次記憶查詢均回傳空清單，不能將它們視為成功取回這 5 筆記憶的證據。

簽章、精確時間、原始部署資訊及重複的 rawEvent 已移除；識別碼、圖片路徑與示範連結已替換。保留人工確認、同意回應及模擬結果之間的對應關係。這是供閱讀的示範資料，無法用來還原原始雲端資料；私人備份不屬於此資料夾。
