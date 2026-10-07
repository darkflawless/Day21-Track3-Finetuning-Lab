# Lab 21 — Evaluation Report

**Họ tên**: Học viên AICB  **MSSV**: AICB-K4-2026  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 30 steps (max_steps = 30) |

**Template có giữ khối `<think>` không?** Có — kiểm tra `results/template_check.json` cho thấy cả thẻ mở `<think>` và nội dung suy luận đều được bảo toàn đầy đủ (`open_tag_present: true`, `body_present: true`), template render đúng định dạng ChatML của dòng Qwen3.5:
```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phân tích: Toàn bộ system prompt và user prompt đều được gán `loss = -100` (masked out hoàn toàn). Chỉ có token câu trả lời của assistant sau thẻ đóng suy luận `</think>` mới được tính gradient (`supervised_fraction = 41.49% < 95%`), đảm bảo mô hình chỉ học dự đoán câu trả lời định dạng JSON mà không học vẹt lại prompt của người dùng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 2891.9 |
| (b) base + optimized prompt | 0.7550 | 0.7911 | 1.0000 | 751.6 |
| (c) LoRA fine-tune | 0.9800 | 0.1778 | 1.0000 | 1284.0 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) vượt trội hoàn toàn so với (a): độ chính xác target tăng từ 0.0000 lên 0.7550, tỷ lệ format chuẩn JSON tăng từ 0.0 lên 1.0 (100% parse được và đủ 4 khóa), đồng thời giảm độ trễ từ 2891.9 ms xuống 751.6 ms (nhanh hơn gần 4 lần) do prompt tối ưu ép mô hình sinh trực tiếp JSON mà không sinh lời dẫn lan man.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không, giữ nguyên prompt tối ưu gốc được chuẩn hóa (SHA-256: `719e74d3b6232053`) để đảm bảo tính liêm chính học thuật và tính công bằng tuyệt đối cho phép đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.3510 | 0.9800 | 386.1 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.3522 | 0.9800 | 249.3 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.4962 | 0.0000 | 376.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.3548 | 0.9650 | 426.0 | 3.55 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Run `attn_only` đã được khớp ngân sách tham số chính xác với `correct` (~32.46M tham số bằng cách nâng rank lên $r=283$, sai lệch chỉ 0.025%). Trên tập target đánh giá đầy đủ 50 mẫu, `attn_only` đạt kết quả hoà tuyệt đối với `correct` (cùng đạt 0.9800), tuy nhiên train loss của nó lại cao hơn một chút so với `correct` (0.3522 so với 0.3510). Điều này cho thấy thứ tự theo train loss không phản ánh trọn vẹn năng lực downstream thực tế của mô hình. Quan trọng hơn, kết quả này chứng minh nguyên lý "LoRA Without Regret" (Thinking Machines 2025): việc phân bổ rank nhỏ ($r=16$) trên toàn bộ các lớp tuyến tính (`text-linear`, bao gồm cả MLP/gate/up/down) mang lại hiệu quả tương đương việc dồn toàn bộ ngân sách tham số với rank khổng lồ ($r=283$) chỉ vào 2 ma trận attention ($q, v$). Vị trí gắn adapter (`placement`) bao quát hơn mới là yếu tố quyết định khả năng biểu diễn tri thức, chứ không phải bản thân giá trị `rank`.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ thay đổi duy nhất một siêu tham số: áp dụng mức learning rate truyền thống của full-finetuning ($\text{LR} = 10^{-5}$ thay vì $10^{-4}$). Kết quả là đường loss của `wrong_lr` gần như đi ngang phẳng lì và kẹt cứng ở mức 1.4962 (trong khi `correct` hội tụ nhanh xuống 0.3510), dẫn tới việc mô hình hoàn toàn không học được tác vụ mục tiêu (target = 0.0000, format = 0.0000). Nếu chỉ quan sát đường loss phẳng mà không biết việc cấu hình sai thang đo learning rate, người làm thực nghiệm rất dễ rơi vào bẫy kết luận sai lầm: cho rằng dữ liệu phân loại quá khó, mô hình Qwen3.5-4B không đủ năng lực xử lý, hoặc LoRA không thể áp dụng cho bài toán này. Đây là bài học thực nghiệm cốt lõi: LoRA cập nhật trên không gian con rank thấp nên gradient bị co hẹp, bắt buộc phải dùng learning rate lớn hơn full-FT từ 5 đến 10 lần ($10^{-4}$) thì trọng số mới có thể di chuyển ra khỏi điểm khởi tạo ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Run `qlora` giúp cắt giảm lượng VRAM đỉnh cực kỳ ấn tượng từ 8.78 GB xuống chỉ còn 3.55 GB (tiết kiệm đến 59.6% bộ nhớ đồ họa, cho phép chạy được cả trên các GPU tiêu dùng dung lượng nhỏ). Tuy nhiên, sự đánh đổi là rất rõ ràng: thời gian huấn luyện bị kéo dài từ 386.1s lên 426.0s (tăng 10.3%) do chi phí giải lượng tử hóa (dequantization) liên tục trong forward pass, và điểm target downstream bị tụt giảm từ 0.9800 xuống 0.9650. Số liệu đo đạc thực nghiệm hoàn toàn ủng hộ khuyến nghị chính thức từ nhà sản xuất (Qwen team): nếu hạ tầng phần cứng đã đáp ứng đủ (như GPU T4 16GB có thể chạy thoải mái 16-bit bf16 với 8.78 GB VRAM), thì không nên dùng QLoRA vì lượng tử hóa 4-bit gây mất mát thông tin và suy giảm thông lượng tính toán mà không mang lại thêm giá trị thực tiễn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.2250` · `regression Δ = -0.6133` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Cổng hồi quy đánh giá đạt phán quyết `FAILED` do vi phạm nghiêm trọng ngưỡng suy thoái năng lực tổng quát (`general capability regressed by 0.613`, vượt xa mức dung sai cho phép là 0.020). Cụ thể, trên 50 mẫu đánh giá mục tiêu phân loại ticket CSKH, mô hình fine-tune LoRA cho thấy sự vượt trội đáng kể (`target Δ = +0.2250`, nâng độ chính xác từ 0.7550 lên 0.9800 so với prompt tối ưu). Tuy nhiên, trên tập kiểm thử hồi quy 15 câu hỏi phổ thông, điểm năng lực tổng quát đã sụt giảm thảm hại từ 0.7911 xuống chỉ còn 0.1778.

Nguyên nhân căn bản là hiện tượng **quên thảm họa (catastrophic forgetting)** và **suy thoái chuỗi lập luận (reasoning-trace collapse)**. Khi fine-tune 30 step trên tập dữ liệu chuyên biệt chỉ chứa ticket CSKH tiếng Việt mà hoàn toàn không có dữ liệu đối ứng đa nhiệm (replay data), trọng số adapter đã ép mô hình rơi vào "thiên kiến định dạng" (format bias). Hơn nữa, việc huấn luyện với `MASK_MODE=assistant-only` trên mô hình có cơ chế suy nghĩ (thinking model) đã triệt tiêu hoàn toàn khả năng sinh reasoning trace (`valid_trace_rate = 0.0`). Khi nhận các câu hỏi phổ thông, mô hình không còn khả năng suy luận mở mà có xu hướng ép về cấu trúc JSON hoặc đưa ra câu trả lời sai lệch. Phán quyết FAILED này phản ánh trung thực rằng một mô hình chuyên biệt hóa thành công trên tác vụ hẹp vẫn có thể phá hủy hoàn toàn giá trị sử dụng chung nếu không được bảo vệ bằng dữ liệu hồi quy.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket / Câu hỏi (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt." | `intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc` | `intent: doi_tra, urgency: trung_binh, product: chuột không dây, sentiment: tich_cuc` | `intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc` | ✅ **FT thắng**: Bắt chuẩn xác độ khẩn cấp `urgency: cao` từ từ khóa "Gấp", trong khi prompt (b) đánh giá nhầm thành `trung_binh`. |
| 2 | "Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình." | `intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc` | `intent: doi_tra, urgency: cao, product: ốp lưng điện thoại, sentiment: tieu_cuc` | `intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc` | ✅ **FT thắng**: Phân biệt chuẩn xác giữa `hoan_tien` và `doi_tra`, prompt (b) bị nhầm do chưa nắm sâu ngữ cảnh đơn hàng. |
| 3 | "Cho mình hỏi, mình đặt chuột không dây mã đơn OD538419. Hoàn tiền. Mong shop phản hồi. Mình vẫn tin tưởng shop." *(Mẫu #9)* | `intent: hoan_tien, urgency: trung_binh, product: chuột không dây, sentiment: tich_cuc` | `intent: hoan_tien, urgency: trung_binh, product: chuột không dây, sentiment: tich_cuc` | `intent: hoan_tien, urgency: thap, product: chuột không dây, sentiment: tich_cuc` | ❌ **FT thua**: FT đoán nhầm `urgency: thap` thay vì `trung_binh` do cụm từ "Mong shop phản hồi" trong ngữ cảnh tin tưởng bị mô hình hạ thấp độ khẩn cấp. |
| 4 | "Cho mình hỏi, mình đặt máy xay sinh tố mã đơn OD906403. Hoàn lại. Mong shop phản hồi. Shop hỗ trợ tốt." *(Mẫu #32)* | `intent: doi_tra, urgency: trung_binh, product: máy xay sinh tố, sentiment: tich_cuc` | `intent: doi_tra, urgency: trung_binh, product: máy xay sinh tố, sentiment: tich_cuc` | `intent: doi_tra, urgency: thap, product: máy xay sinh tố, sentiment: tich_cuc` | ❌ **FT thua**: Tiếp tục sai lệch ở trường `urgency` (đoán `thap` thay vì `trung_binh`), cho thấy mô hình FT có xu hướng dồn xác suất về nhãn đa số. |
| 5 | "Thủ đô của Việt Nam là thành phố nào?" *(Câu hỏi kiểm tra hồi quy tri thức)* | Từ khóa: `Hà Nội` | "Thủ đô của Việt Nam là thành phố Hà Nội." | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "none", "sentiment": "trung_tinh"}` | ❌ **FT thua thảm hại**: Mô hình FT bị format bias cực đoan, tự động biến một câu hỏi địa lý thông thường thành JSON ticket CSKH và đánh mất câu trả lời factual. |

**Có mẫu chung nào ở các ca FT thua không?**
Phân tích các ca FT thua cho thấy hai mẫu số chung rõ ràng:
1. **Trên tác vụ mục tiêu:** Mô hình FT bị thiên lệch nhẹ ở trường `urgency` đối với các câu có sắc thái nhẹ nhàng ("Mong shop phản hồi", "tin tưởng shop") $\rightarrow$ mô hình có xu hướng dự đoán nhãn phổ biến nhất (`thap`) thay vì nắm bắt đúng quy ước gán nhãn `trung_binh` của chuyên gia dữ liệu.
2. **Trên tác vụ ngoài miền (OOD / Regression):** Mô hình bị "ám ảnh cấu trúc" (format obsession), mất khả năng đàm thoại tự nhiên và cố gắng ép mọi câu hỏi tri thức mở vào khuôn khổ JSON 4 trường của bài toán ticket CSKH.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**
Kết quả đo đạc từ toàn bộ 50 mẫu đánh giá thực nghiệm đưa ra câu trả lời dứt khoát: **Chưa thể đưa bản fine-tune LoRA này vào môi trường production.**

Mặc dù bản fine-tune đạt độ chính xác tác vụ mục tiêu vượt trội ($0.9800$, cao hơn $22.5\%$ so với baseline prompt tối ưu và đảm bảo cấu trúc JSON hợp lệ $100\%$), nhưng nó đã thất bại toàn diện trước cổng kiểm định an toàn hồi quy. Mô hình đánh mất $61.3\%$ độ chính xác trên tri thức tổng quát (tụt từ $0.7911$ xuống $0.1778$) và làm sụp đổ hoàn toàn khả năng tư duy suy luận (`valid_trace_rate = 0.00`). Hơn nữa, độ trễ suy luận của bản fine-tune ($1284.0\text{ ms}$) cao hơn $1.7$ lần so với việc dùng prompt tối ưu trên base model ($751.6\text{ ms}$), làm tăng đáng kể chi phí hạ tầng khi phục vụ ở quy mô lớn.

Hệ thống thí nghiệm đối chứng trong lab đã chỉ ra các đòn bẩy kỹ thuật mang tính quyết định:
1. **Dữ liệu và cơ chế bảo tồn hồi quy là đòn bẩy tối cao:** Huấn luyện thuần túy trên miền hẹp mà không trộn $1–5\%$ dữ liệu tổng quát (replay buffer) chắc chắn dẫn đến quên thảm họa.
2. **Vị trí gắn adapter (`placement`) quan trọng hơn độ lớn của `rank`:** Cấu hình dàn đều tất cả các khối tuyến tính (`text-linear`) với rank nhỏ $r=16$ đạt độ chính xác ngang bằng cấu hình nhồi nhét $r=283$ vào attention (`attn_only`), đồng thời giữ được tính tổng quát tốt hơn.
3. **Learning Rate phải phù hợp với bản chất LoRA:** Sử dụng LR thang full-FT ($10^{-5}$) sẽ khiến quá trình học sụp đổ hoàn toàn.

**Ba điều tôi học được:**
1. **Đừng bao giờ đánh giá mô hình bằng một chỉ số đơn lẻ:** Một mô hình đạt accuracy 98% trên tập test chuyên biệt vẫn có thể là một mô hình "hỏng" nếu nó bị phá hủy tri thức nền tảng và tư duy lập luận. Cổng hồi quy 4 nhóm là công cụ bắt buộc để bảo vệ chất lượng mô hình trong thực tế.
2. **Loss Masking là nền móng của sự liêm chính:** Chứng minh mask ngược bằng code giúp khẳng định gradient chỉ chảy qua token câu trả lời. Nếu tính loss cả trên prompt, mô hình chỉ đơn thuần học vẹt hình thức mà không thực sự khái quát hóa bài toán.
3. **Prompt Engineering là một baseline cực kỳ đáng gờm:** Chỉ bằng việc chuẩn hóa ChatML và tối ưu prompt chỉ dẫn, baseline (b) đã đạt accuracy 75.5%, format 100% với tốc độ nhanh nhất ($751.6\text{ ms}$) mà không tốn một giây huấn luyện nào. Fine-tuning chỉ nên là bước tiếp theo khi prompt đã chạm giới hạn kỹ thuật.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. **Trộn 3–5% dữ liệu tổng quát (Instruction Replay):** Đưa thêm tập dữ liệu hội thoại và kiến thức phổ thông vào `train_seed.jsonl` để duy trì năng lực nền tảng, giúp bản fine-tune vượt qua cổng hồi quy (PASS).
2. **Thử nghiệm `MASK_MODE=response-only` trên chuỗi suy luận:** Chỉ tính loss trên câu trả lời sau thẻ `</think>` nhưng giữ nguyên việc mô hình sinh reasoning trace ở forward pass, nhằm bảo toàn chuỗi tư duy và cải thiện `valid_trace_rate` theo hướng dẫn tại Deck §17.5.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
