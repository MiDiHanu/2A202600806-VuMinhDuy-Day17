# Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, bài tập tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, em xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread.
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài.

Mục tiêu cuối cùng không phải chỉ là "agent nhớ nhiều hơn", mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Cấu trúc repo

```
.
├── README.md
├── Guide.md
├── Rubric.md
├── data/
│   ├── conversations.json          # 5 conversation ngắn (recall có correction)
│   └── advanced_long_context.json  # 1 stress test 15 turn rất dài
├── src/
│   ├── config.py
│   ├── model_provider.py
│   ├── memory_store.py
│   ├── agent_baseline.py
│   ├── agent_advanced.py
│   ├── benchmark.py
│   ├── test_agents.py
│   └── README.md
└── state/                          # sinh ra khi chạy benchmark (User.md, profiles)
```

## Cách chạy

```bash
# cài tối thiểu (đã có sẵn trên máy em)
python -m pip install pytest python-dotenv

# chạy test
cd src
python -m pytest test_agents.py -v

# chạy benchmark
cd src
python benchmark.py
```

Benchmark chạy hoàn toàn offline (deterministic), không cần API key. Nếu muốn thử live mode với provider thật, set biến môi trường tương ứng (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) trước khi chạy.

## Provider hỗ trợ

- `openai`
- `custom` (OpenAI-compatible base URL — vd LM Studio, vLLM)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Nếu thiếu SDK tương ứng, agent tự fallback về offline mode thay vì crash.

## Kết quả benchmark (offline)

### Standard benchmark (`data/conversations.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 1733 | 16,623 | 0.07 | 0.44 | 0 | 0 |
| Advanced | 1326 | 21,471 | 0.32 | 0.60 | 368 | 0 |

### Long-context stress benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 302 | 22,564 | 0.00 | 0.40 | 0 | 0 |
| Advanced | 235 | 14,214 | 0.50 | 0.70 | 236 | 3 |

## Phân tích

### 1. Tại sao advanced có recall tốt hơn baseline?

Advanced lưu fact dài hạn vào `User.md` (1 file markdown / user). Mỗi khi cần trả lời, agent đọc lại file này và tìm đúng key (name, location, profession, favorite_drink, ...). Baseline chỉ nhớ trong thread, nên khi recall question hỏi ở một thread *mới*, agent không có dữ liệu để trả lời — recall gần như bằng 0 ở long-context stress (0.00), trong khi advanced đạt 0.50.

### 2. Tại sao advanced có thể tốn hơn ở hội thoại ngắn?

Mỗi lượt của advanced phải mang theo `User.md` + summary + recent messages trong prompt. Ở hội thoại ngắn (`conversations.json`), phần này là overhead thuần túy vì tổng ngữ cảnh vốn đã nhỏ. Kết quả là:

- Standard: advanced tốn 21,471 prompt tokens, nhiều hơn baseline (16,623) ~29%.
- Ở hội thoại ngắn, baseline thắng về chi phí.

### 3. Tại sao compact giúp advanced có lợi thế ở hội thoại dài?

Khi thread vượt ngưỡng `compact_threshold_tokens` (mặc định 1,200), `CompactMemoryManager` sẽ tóm tắt phần message cũ và chỉ giữ lại `keep_messages` mới nhất. Ở stress benchmark:

- Baseline: 22,564 prompt tokens — phải kéo *toàn bộ* lịch sử.
- Advanced: 14,214 prompt tokens — compact đã nén 3 lần.

→ Compact chủ yếu tối ưu `prompt tokens processed` (giảm ~37%), không phải `agent tokens only` (lượng agent sinh ra gần như nhau).

### 4. Memory growth và rủi ro

- File `User.md` chỉ phình khi có fact mới (cùng value sẽ bị idempotent skip).
- Ở stress test, advanced chỉ tốn 236 bytes — vì phần lớn thread là news dài, không phải fact có cấu trúc.
- Rủi ro cần để ý:
  - **Phình to**: Nếu user thường xuyên cung cấp fact mới, `User.md` sẽ tăng. Cần chiến lược dọn (memory decay, archive old facts).
  - **Lưu sai fact**: Khi user vô tình nói "đùa đó, mình không phải PM", nếu regex không có noise filter, fact sẽ bị ghi đè. Code hiện tại đã chặn các cụm "đùa", "chỉ là", "không phải" trên turn ngắn.
  - **Conflict**: Nếu user correction (vd "giờ mình ở Huế chứ không còn ở Đà Nẵng"), `upsert_fact` thay thế bullet cũ thay vì tạo bullet mới trùng key.

### 5. Tại sao compact không phải lúc nào cũng thắng ở hội thoại ngắn?

Vì summary tự nó tốn tokens. Ở thread ngắn, cộng thêm overhead đọc `User.md` thì tổng `prompt tokens processed` lại *tăng*. Compact chỉ thật sự có lợi khi tổng số token cũ đủ lớn để việc nén + giữ summary + recent messages < việc giữ nguyên toàn bộ lịch sử.

## Cấu trúc kỹ thuật của hệ thống memory

Hệ thống có 3 lớp tách bạch trong `src/agent_advanced.py`:

1. **Short-term** (in-thread): được đẩy vào `CompactMemoryManager`.
2. **Persistent** (`User.md`): từng fact được lưu 1 bullet markdown; idempotent nếu cùng value; replace nếu value mới.
3. **Compact**: khi `total_tokens >= threshold` thì tóm tắt phần cũ + giữ `keep_messages` gần nhất; đếm số compaction để benchmark.

Mỗi lớp có API riêng (`profile_store.*`, `compact_memory.*`) nên dễ test và dễ thay thế.

## Bonus đã triển khai (90-100)

- **Confidence threshold** (`extract_profile_updates`): bỏ qua turn là câu hỏi thuần, lọc nhiễu ("đùa", "chỉ là", "không phải"), bỏ value rỗng / là question word ("gì", "đâu", "nào").
- **Conflict handling** (`_store_fact`): khi fact mới mâu thuẫn fact cũ, `upsert_fact` thay thế thay vì nhân đôi bullet.
- **Entity extraction có cấu trúc**: tách hẳn các field (name, location, profession, response_style, favorite_drink, favorite_food, pet, interests) với regex riêng.

### Rủi ro của bonus

- Regex chỉ là heuristic — chưa cover tiếng Việt đầy đủ các biến thể cú pháp. Khi scale lên production, cần thay bằng LLM-based extraction hoặc schema có guardrail tốt hơn.
- Conflict handling hiện chỉ là "last write wins". Nếu user correction lẫn lộn với nhiễu (vd "mình đùa đó, MLOps engineer chứ"), cần thêm confidence score.

## Tài liệu tham khảo

- `Guide.md`: hướng dẫn từng bước của lab.
- `Rubric.md`: tiêu chí chấm điểm chi tiết.
- `src/README.md`: mô tả scaffold gốc.
