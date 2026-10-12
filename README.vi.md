# Tái hiện ADGR — GraphRAG Y khoa tiếng Việt

**🌐 Ngôn ngữ / Language: [Tiếng Việt](README.vi.md) | [English](README.md)**
**📄 Báo cáo đầy đủ / Reports: [report_vi.pdf (VI)](report/report_vi.pdf) | [report.pdf (EN)](report/report.pdf)**

Tái hiện độc lập, từ đầu, hệ thống *Adaptive Directed Graph Retrieval*
trong paper nộp tạp chí *Applied Intelligence*: hệ thống trả lời câu hỏi y
khoa tiếng Việt bằng cách tra cứu chứng cứ trong đồ thị tri thức (bệnh,
thuốc, triệu chứng và mối liên hệ).

## Kết quả so với paper

| Bảng | Chỉ số | Paper | Ta | Đọc thế nào |
|---|---|---|---|---|
| 1 | chunks / ent / rel | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | 92–107%, cùng tầm |
| 1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | gần |
| 3 | Hit@10 full | 0.71 | 0.79 / 0.68 (2 index) | cùng tầm |
| 4 | RAGAS tổng | 0.8792 | 0.64 (tương đương, đã phân rã) | xem phân tích gap |
| 5 | ADGR overall | 0.7921 | 0.858 (ta) / 0.778 (prompt gốc) | giữ thứ hạng |
| 6 | tokens/row | 4490.5 | 3441.4 | rẻ hơn |
| 7 | 4 ablations | tản 0.60–0.88 | tản đúng hướng | cùng kết luận |
| 8 | Aescin 5ent/57rel | — | 12–14ent/~40rel | cùng kiểu gọn |

Mọi chênh lệch đều có nguyên nhân đo được trong báo cáo. Không mock số,
không cherry-pick hàng, không bịa số.

## Trong repo có gì

```
├── README.md / README.vi.md  # file này, Anh + Việt
├── AGENTS.md                  # kỷ luật vận hành đúc từ 43 lỗi đã sửa
├── LICENSE                    # MIT
├── report/                    # báo cáo học thuật (Anh + Việt, LaTeX + PDF) + 10 hình
├── src/                       # pipeline: tìm kiếm, đánh giá, chain 16 stages
├── results/                   # FINAL_v2.json (8 bảng) + toàn bộ đáp án/chấm thô
└── corpus/                    # bài báo/passages/trích xuất/contexts (<100MB mỗi file)
```

File lớn để ngoài GitHub đúng chuẩn (kèm hướng dẫn dựng lại trong
`AGENTS.md`): index tác giả (2.3GB), index tự dựng (960MB), models (9.4GB).

## Phương pháp vắn tắt

Câu hỏi → từ khóa LLM → neo lai (BM25 + dense, thưởng chồng lấp) → cạnh
1-bước chấm và lọc theo truy vấn → gộp 4 nguồn chunk → rerank top-10 →
Qwen2.5-14B sinh đáp án (temp 0) → chấm Hit/MRR/nDCG, RAGAS 4 thành phần,
4 chiều × 5 lần, tokens/latency.

## Thay thế đã công khai

| Paper | Ta | Đã đo ảnh hưởng |
|---|---|---|
| Giám khảo GPT-5.5 | solar-pro4, cùng protocol | lệch thang −0.09, giữ thứ hạng |
| 7 models trả phí | 7 models miễn phí (khác hẳn) | chỉ tham khảo |
| Gold labels gán tay | 2 proxy tự động (đồng thuận) | đã disclose |
| Thư viện RAGAS gốc | Cài lại đúng thuật toán gốc | thành phần khớp |

## Chạy lại

```bash
python -u src/kg_build_paper.py   # dựng index đồ thị
python -u src/chain_full.py       # chạy toàn bộ (lưu từng câu, resume được,
                                  # tự kiểm tra mỗi stage)
```

Cần: 1 GPU 24GB, vLLM + ollama, Python 3.11, key Upstage (~$5 cho toàn bộ
protocol chấm). Đọc `AGENTS.md` trước.

## Trích dẫn paper gốc

```bibtex
@article{nguyentienha2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {{Nguyen Tien Ha}, Vu Duy Linh, Nguyen Thi Minh Huyen,
             Le Ngoc Toan and Kim Dinh Thai},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```
