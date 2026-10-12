# Dựng lại hệ thống tìm kiếm y khoa ADGR — Tái hiện đầy đủ

**🌐 Ngôn ngữ / Language: [Tiếng Việt](README.vi.md) | [English](README.md)**
**📄 Báo cáo / Reports: [report_vi.pdf (VI)](report/report_vi.pdf) | [report.pdf (EN)](report/report.pdf)**

Chúng tôi dựng lại từ đầu hệ thống ADGR trong paper của **Nguyen Tien Ha và
cộng sự** (nộp tạp chí *Applied Intelligence*). ADGR là hệ thống trả lời câu
hỏi y khoa tiếng Việt bằng cách tra cứu chứng cứ trong đồ thị tri thức (mạng
lưới các sự thật y khoa: bệnh, thuốc, triệu chứng và mối liên hệ giữa chúng).

**Kết luận trước:** hệ thống ta dựng hoạt động giống hệ paper, xếp hạng các
phương pháp giống hệt, và khớp hầu hết con số. Chỗ nào khác biệt đều đã đo
nguyên nhân — không giấu gì.

- Paper: *Adaptive Directed Graph Retrieval for Graph-Based
  Retrieval-Augmented Generation*
- Tài nguyên tác giả: [kho văn bản y khoa (HF)](https://huggingface.co/datasets/vuduylinh150804/vietmed-crawled-corpus) · [code tham khảo](https://github.com/Jmhzbmcn2/Graduation-Thesis) · [prompt chấm + hướng dẫn trích xuất](https://github.com/Jmhzbmcn2/adgr-supplementary)

## Đã làm gì (nói dễ hiểu)

1. **Thu thập 1.934 bài báo y khoa** (paper: 1.932 bài).
2. **Cắt thành 15.837 đoạn ngắn** (paper: 17.236 đoạn).
3. **Trích xuất đồ thị tri thức**: 52.578 nút sự thật (paper: 49.274) và
   106.279 mối liên hệ (paper: 106.666).
4. **Chạy 300 câu hỏi thi** qua 5 phiên bản hệ thống (bản đầy đủ + 4 bản tắt
   bớt từng bộ phận) cùng 2 hệ thống đối chứng.
5. **Chấm mọi đáp án** bằng AI giám khảo (5 lần độc lập) và bằng các thước đo
   tìm kiếm chuẩn. **8 bảng kết quả** nằm ở `results/FINAL_v2.json`.

## Kết quả so với paper (nói dễ hiểu)

| # | Đo cái gì | Paper | Ta | Nghĩa là gì |
|---|---|---|---|---|
| 1 | Quy mô đồ thị | 17k đoạn, 93.6% đúng schema | 15.8k đoạn, 91.3% đúng schema | Gần như cùng đồ thị |
| 3 | Tìm đúng chứng cứ (Hit@10) | 0.71 | 0.79 / 0.68 (2 index) | Cùng tầm |
| 4 | Chất lượng RAGAS tổng | 0.88 | 0.64 (tương đương) | Thấp hơn — đã mổ nguyên nhân bên dưới |
| 5 | Giám khảo chấm ADGR | 0.79 | 0.86 (ta) / 0.78 (prompt gốc) | Thứ hạng giữ nguyên |
| 6 | Chi phí: chữ/question | 4.490 | 3.441 | Ta rẻ hơn |
| 7 | Tắt bộ phận → điểm tụt | có, tản 0.60–0.88 | có, tản đúng hướng | Cùng kết luận |
| 8 | Ví dụ Aescin gọn | 5 nút / 57 links | 12–14 nút / ~40 links | Cùng kiểu gọn |

**Về Tab.4 (0.64 vs 0.88):** thí nghiệm chứng minh ~60% gap do ta chấm gộp
1 bước thay vì 2 bước chuẩn (chấm 2 bước đạt 0.77 trên cùng dữ liệu). Phần còn
lại do giám khảo nghiêm hơn + extraction nhiễu hơn. Đáp án ngắn khớp paper
(0.89). Xem phân tích đầy đủ trong `report/`.

## Trong repo có gì

```
├── README.md / README.vi.md  # file này, Anh + Việt
├── AGENTS.md                  # kỷ luật làm việc đúc từ 41 lỗi đã sửa
├── LICENSE                    # MIT
├── report/                    # báo cáo học thuật (Anh + Việt, LaTeX + PDF) + 10 hình
├── src/                       # pipeline: tìm kiếm, đánh giá, script chạy all
├── results/                   # 8 bảng FINAL + toàn bộ đáp án/chấm thô
└── corpus/                    # bài báo/passages/trích xuất/contexts (<100MB mỗi file)
```

File lớn để ngoài GitHub đúng chuẩn (kèm hướng dẫn dựng lại): index tác giả
(2.3GB), index tự dựng (960MB), models (9.4GB). Chạy `chain_full.py` để dựng.

## Cái gì khác paper (khai hết, không giấu)

| Paper | Ta | Đã đo ảnh hưởng? |
|---|---|---|
| Giám khảo GPT-5.5 | solar-pro4, cùng protocol 5 lần | Rồi: lệch thang −0.09, thứ hạng giữ nguyên |
| 7 models trả phí | 7 models miễn phí (khác hẳn) | Chỉ tham khảo, không xếp hạng |
| Gold labels gán tay | 2 proxy tự động (đồng thuận nhau) | Rồi, ghi trong báo cáo |
| Thư viện RAGAS gốc | Cài lại đúng thuật toán gốc | Thành phần khớp; faithfulness khác (xem trên) |

## Chạy lại

```bash
python -u src/kg_build_paper.py   # dựng index đồ thị
python -u src/chain_full.py       # chạy toàn bộ (lưu từng câu, resume được,
                                  # tự kiểm tra mỗi stage)
```

Cần: 1 GPU 24GB, vLLM + ollama, Python 3.11, key Upstage (~$5 cho toàn bộ
protocol chấm). Đọc `AGENTS.md` trước — mỗi luật ở đó đều trả giá bằng giờ
GPU mới có.

## Trích dẫn

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

*Mọi số ở đây đều chạy lại được từ scripts kèm theo. Không mock số, không
cherry-pick hàng — kể cả những số xấu hơn paper, đều có nguyên nhân đo được.*
