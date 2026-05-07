# Lab 21 — Evaluation Report

**Học viên**: <Họ tên> — <MSSV>  
**Ngày nộp**: 2026-05-07  
**Submission option**: C (code-only)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: ~$0.07 (~12.2 phút cho cả 3 phiên experiment @ $0.35/hr)
- **HF Hub link**: (N/A)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.0 min    | 7.2 GB    | 1.5577    | 4.75       |
| 16   | 3,686,400       | 4.3 min    | 6.6 GB    | 1.5161    | 4.55       |
| 64   | 14,745,600      | 4.0 min    | 8.0 GB    | 1.4768    | 4.38       |
| Base | -               | -          | -         | -         | -          |

*Lưu ý: Peak VRAM của r=16 thấp hơn r=8 có thể do cách theo dõi bộ nhớ của PyTorch hoặc tình trạng phân mảnh tại thời điểm chạy.*

## 3. Loss Curve Analysis
- **Quan sát**: Đường cong training loss giảm đều từ khoảng 1.6 xuống 1.38 sau 3 epochs. Không có dấu hiệu overfitting rõ rệt do loss vẫn đang trên đà giảm và khoảng cách giữa các bước logging khá ổn định. Tuy nhiên, do tắt eval-during-training trên T4 để tiết kiệm VRAM nên chỉ quan sát được training loss.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu...  
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp...  
**Nhận xét**: **Improved**. Câu trả lời súc tích và đi thẳng vào trọng tâm hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Cung cấp hàm đệ quy cơ bản.  
**Fine-tuned (r=16)**: Cung cấp hàm có thêm kiểm tra điều kiện đầu vào (`ValueError` nếu n âm), giúp code robust hơn.  
**Nhận xét**: **Improved**.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Thân thiện người dùng, sắp xếp bố cục, màu sắc...  
**Fine-tuned (r=16)**: Chuyển đổi, Thích ứng, Đơn giản, Nhất quán...  
**Nhận xét**: **Improved**. Các nguyên tắc được liệt kê mang tính chuyên môn cao hơn.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Giải thích đúng LoRA là Low-Rank Adaptation.  
**Fine-tuned (r=16)**: Giải thích sai tên viết tắt LoRA thành "Layer-wise Adaptive Regularization Optimization" dù nội dung kỹ thuật vẫn có phần đúng.  
**Nhận xét**: **Degraded**. Xuất hiện hiện tượng ảo giác (hallucination) về thuật ngữ viết tắt.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Ba cách khác nhau để cải thiện hiệu suất...  
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI...  
**Nhận xét**: **Same**. Cả hai đều đưa ra định nghĩa chính xác.

## 5. Conclusion về Rank Trade-off

Dựa trên dataset Vietnamese Alpaca này:
- **Rank cho ROI tốt nhất**: **r=16** là sự lựa chọn cân bằng nhất. Mặc dù r=64 cho Perplexity thấp nhất (4.38), nhưng chi phí tài nguyên (VRAM tăng lên 8.0GB) và rủi ro Overfitting cao hơn trên tập dữ liệu nhỏ (200 samples). r=16 cải thiện rõ rệt so với r=8 mà không tốn quá nhiều tài nguyên.
- **Diminishing returns**: Khi tăng rank từ 16 lên 64 (gấp 4 lần tham số trainable), Perplexity chỉ giảm từ 4.55 xuống 4.38 (khoảng 3.7%). Điều này cho thấy việc tăng rank quá lớn trên một dataset nhỏ không mang lại hiệu quả đột phá.
- **Recommendation**: Nếu deploy production, tôi chọn **rank 16**. Lý do là nó đủ để mô hình nắm bắt được phong cách trả lời mới (fine-tuning style) mà vẫn giữ được tính gọn nhẹ của adapter, giúp inference nhanh hơn và tiết kiệm bộ nhớ khi quản lý nhiều adapter cùng lúc.

## 6. What I Learned
- **Sử dụng Unsloth**: Giúp quá trình fine-tuning trên GPU hạn chế như T4 trở nên khả thi và nhanh chóng (chỉ mất vài phút cho mỗi experiment).
- **Hiện tượng Hallucination**: Fine-tuning với rank cao hoặc trên dữ liệu ít có thể khiến mô hình tự tin trả lời sai các chi tiết nhỏ (như giải thích sai tên viết tắt LoRA).
- **Cấu hình Rank**: Việc chọn rank không phải cứ cao là tốt, nó phụ thuộc mật thiết vào độ khó của task và kích thước của dataset.
