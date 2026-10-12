# Tái hiện ADGR — GraphRAG Y khoa tiếng Việt

Bản tiếng Việt của [README.md](README.md) · Vietnamese version of the English README.

Tái hiện độc lập, từ đầu, framework **Adaptive Directed Graph Retrieval
(ADGR)** (Nguyen-Tien et al., nộp *Applied Intelligence*) cùng đồ thị tri thức
y khoa tiếng Việt — mọi con số đều công khai nguồn gốc, không mock, không
cherry-pick.

- Paper gốc: *Adaptive Directed Graph Retrieval for Graph-Based
  Retrieval-Augmented Generation*
- Tài nguyên tác giả: [ViMedGraph (HF)](https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus) · [code tham khảo](https://github.com/Jmhzbmcn2/Graduation-Thesis) · [prompt judge gốc (Online Resource 1)](https://github.com/Jmhzbmcn2/adgr-supplementary)

## Kết quả tổng quan (ta vs paper)

| Bảng | Chỉ số | Paper | Ta | Nhận xét |
|---|---|---|---|---|
| Tab.1 | chunks / ent / rel | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | 92–107% |
| Tab.1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | gần |
| Tab.3 | Hit@10 full | 0.71 | 0.79 / 0.68 (2 index) | cùng tầm |
| Tab.4 | RAGAS overall | 0.8792 | 0.64 (tương đương) | xem phân rã gap |
| Tab.5 | ADGR overall | 0.7921 | 0.858 (ta) / 0.778 (prompt gốc) | thứ hạng giữ |
| Tab.6 | tokens/row | 4490.5 | 3441.4 | rẻ hơn |
| Tab.7 | 4 ablations | tản 0.60–0.88 | tản đúng hướng | cơ chế đúng |
| Tab.8 | Aescin 5ent/57rel | — | 12–14ent/39–41rel | cùng tầm gọn |

Gap faithfulness (0.41 vs 0.89) đã phân rã bằng thí nghiệm: protocol 2-bước
chuẩn đạt 0.77 trên cùng dữ liệu → ~60% gap do shortcut 1-call của ta, còn lại
do judge nghiêm hơn + extraction nhiễu hơn. Chi tiết trong `report/`.

## Cấu trúc repo

```
├── README.md / README.vi.md  # tiếng Anh + tiếng Việt
├── AGENTS.md                  # kỷ luật vận hành đúc từ 41 lỗi đã sửa
├── LICENSE                    # MIT
├── report/                    # báo cáo EN + VI (LaTeX + PDF) + 10 hình
├── src/                       # pipeline: retrieval, eval, chain 16 stages
├── results/                   # FINAL_v2.json (8 bảng) + checkpoints + metrics
└── corpus/                    # crawl/chunks/extraction/contexts (<100MB/file)
```

## Thay thế đã công khai

| Paper | Tái hiện này |
|---|---|
| Judge GPT-5.5 | solar-pro4 (temp 0, 5 runs; + bản prompt gốc) |
| 7 LLM trả phí | 7 models miễn phí (không trùng, chỉ tham khảo) |
| Gold labels thủ công | proxy slug-gold + single-gold (đã đo đồng thuận) |
| Thư viện RAGAS gốc | Cài lại thuật toán gốc + direct-rating faithfulness |

## Tái chạy

```bash
python -u src/kg_build_paper.py   # dựng index (R6-R7)
python -u src/chain_full.py       # toàn chuỗi (checkpoint từng row, gates kiểm tra)
```

Cần: 1 GPU 24GB, vLLM + ollama, Python 3.11, key Upstage (~$5 cho toàn bộ protocol Tab.4/5/7). Xem `AGENTS.md` trước khi chạy.

## Trích dẫn

```bibtex
@article{nguyentien2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {Nguyen-Tien, Ha and Vu-Duy, Linh and Nguyen-Thi, Minh-Huyen
             and Le-Ngoc, Toan and Kim, Dinh-Thai},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```
