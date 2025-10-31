# Local RAG Chat (Streamlit)

Ứng dụng này là một ví dụ Local RAG (Retrieval-Augmented Generation) dùng:
- Streamlit làm UI
- ChromaDB (persistent) làm vector store (lưu ở `data/vector_store`)
- sentence-transformers (`all-MiniLM-L6-v2`) để tạo embedding
- Một mô hình LLM chạy cục bộ qua HTTP (ví dụ Ollama) để sinh câu trả lời

Mục tiêu: người dùng upload 1 file PDF, nội dung được index thành các đoạn văn bản, rồi có thể đặt câu hỏi dựa trên nội dung đã upload. Ứng dụng có tùy chỉnh endpoint/model cho LLM và một nút probe nhanh để debug endpoint.

## Thư mục chính
- `streamlit_app.py` — ứng dụng Streamlit chính (UI, ingest, retrieve, gọi LLM)
- `requirements.txt` — (có sẵn) danh sách phụ thuộc Python
- `data/uploaded_docs/` — nơi lưu file PDF được upload
- `data/vector_store/` — thư mục dữ liệu persistent của ChromaDB

## Yêu cầu (Prerequisites)
- Python 3.8+ (khuyến nghị 3.10/3.11)
- Một môi trường ảo (virtualenv / venv / conda)
- Mô hình LLM chạy cục bộ có endpoint HTTP (ví dụ Ollama). Mặc định app dùng `http://localhost:11434/api/generate` và model mặc định `gemma:2b` nhưng có thể thay đổi trong sidebar.

## Cài đặt (PowerShell)
Mở PowerShell tại thư mục dự án và chạy:

```powershell
# tạo và activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# cài phụ thuộc
pip install -r requirements.txt
```

Nếu bạn dùng conda thay thế, tạo env rồi `pip install -r requirements.txt` trong môi trường đó.

## Chạy ứng dụng
Sau khi cài xong, chạy:

```powershell
streamlit run streamlit_app.py
```

Mở trình duyệt theo URL mà Streamlit cung cấp (thường là `http://localhost:8501`).

## Sử dụng ứng dụng
1. Upload 1 file PDF qua widget `Upload a PDF` — ứng dụng sẽ lưu file vào `data/uploaded_docs/` và index các đoạn văn thành vector store.
2. Ở thanh bên (sidebar):
   - `Model`: thay model (ví dụ `gemma:2b`) nếu bạn dùng Ollama hoặc server khác
   - `LLM endpoint URL`: đổi endpoint nếu server của bạn dùng đường dẫn khác
   - `Test LLM endpoint`: gửi payload kiểm tra nhanh để xem status và body trả về (tiện để debug 404 / lỗi)
3. Gõ câu hỏi vào ô `Ask a question:` và nhấn `Ask`.
   - Ứng dụng sẽ truy vấn vector DB để lấy những đoạn liên quan, xây prompt (bằng tiếng Việt) và gọi LLM.
   - Kết quả hiển thị ở phần `Answer` cùng danh sách `Sources used` (Nguồn đã dùng).

## Cấu hình quan trọng
- `OLLAMA_URL` trong `streamlit_app.py` mặc định là `http://localhost:11434/api/generate`. Nếu server của bạn khác (ví dụ `http://localhost:11434/v1/generate`), hãy đổi trong sidebar hoặc đặt biến môi trường `OLLAMA_URL` trước khi chạy.
- `LLM_MODEL` env var (hoặc ô `Model` trong sidebar) để chọn model mặc định (ví dụ `gemma:2b`).

Dặt biến môi trường trước khi chạy Streamlit:

```powershell
$env:LLM_MODEL = "gemma:2b"
$env:OLLAMA_URL = "http://localhost:11434/api/generate"
streamlit run streamlit_app.py
```

## Debug & Khắc phục sự cố
- Nếu `Test LLM endpoint` trả về `404`:
  - Kiểm tra server LLM (đã bật chưa, đang nghe port 11434 hay port khác)
  - Kiểm tra đường dẫn endpoint đúng chưa (nhiều server dùng `/v1/generate` hoặc endpoint khác)
  - Dùng `Invoke-RestMethod` hoặc `curl` để thử nhanh từ terminal

- Nếu LLM trả `{"response":"...","done":false}` hoặc trả partial chunks:
  - Mình đã thêm `stream: False` và `max_new_tokens` vào payload; nhiều server sẽ trả kết quả hoàn chỉnh khi nhận tham số này.
  - Nếu server vẫn stream, cần dùng API streaming (SSE/websocket) hoặc poll tiếp — xem docs server LLM.

- Nếu ứng dụng không tìm thấy nguồn phù hợp (kết quả rỗng):
  - Kiểm tra file PDF đã được index (app sẽ hiển thị số chunk đã index sau upload)
  - Tăng `TOP_K` trong mã để tìm nhiều candidate hơn

## Gợi ý cải tiến (next steps)
- Thêm tính năng quản lý tài liệu (danh sách file đã index, xóa index cho 1 file, re-index)
- Hỗ trợ streaming responses từ LLM (SSE/websocket) để hiện kết quả dần và hoàn thiện
- Thêm unit tests cho logic `retrieve`/`ingest`
- Thêm logging chi tiết / telemetry cho debug
