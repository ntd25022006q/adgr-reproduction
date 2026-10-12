# Tái hiện ADGR — GraphRAG Y khoa tiếng Việt

**🌐 Ngôn ngữ / Language: [Tiếng Việt](README.vi.md) | [English](README.md)**
**📄 Báo cáo đầy đủ / Reports: [report_vi.pdf (VI)](report/report_vi.pdf) | [report.pdf (EN)](report/report.pdf) — 12 trang mỗi bản, hình nào cũng có hướng dẫn đọc**

Tái hiện độc lập, từ đầu, hệ thống *Adaptive Directed Graph Retrieval*
(paper nộp tạp chí *Applied Intelligence*): trả lời câu hỏi y khoa tiếng Việt
bằng cách tra cứu chứng cứ trong knowledge graph (bệnh, thuốc, triệu chứng và
mối liên hệ).

Nguyên tắc xuyên suốt: **không đòi số giống hệt khi điều kiện khác nhau —
nhưng không có chênh lệch nào thiếu nguyên nhân đã đo.** 9 kết luận
KHỚP/GẦN/VƯỢT, 5 GIẢI THÍCH ĐƯỢC, 0 KHÔNG GIẢI THÍCH ĐƯỢC.

## Quy mô đồ thị: cùng tầm paper (Bảng 1)

<img src="report/figs/fig_tab1_scale.png" width="700">

Dựng lại 52.578 entity / 106.279 relation so với 49.274 / 106.666 của paper
(107% / −0,4%), từ 15.837 chunk (91,9% — 5 URL chết lúc crawl, cứu 4 bằng
nguồn thay thế, công bố hết). Chất lượng tính lại từ extraction thô: DSER
0,913; SCR 0,884; LCR 0,816 (paper: 0,936 / 0,928 / 0,906). Hố sâu nhất là LCR
−9 điểm: chunker H2 to để 18% entity nằm trên đảo tách rời — retrieval vẫn tốt
(Bảng 3) nên cụm lõi đã đủ dùng.

## Tỉ lệ tìm đúng bám paper (Bảng 3)

<img src="report/figs/fig_tab3_hit.png" width="700">

Hit@10 bản đủ: **0,79 / 0,68** trên hai index (paper: 0,71). Thang lạc quan
slug-gold lên 0,94; thang bi quan single-gold 0,79 — số paper nằm trong khoảng
này. Hai proxy độc lập cho cùng thứ tự arm tuyệt đối, nên thứ tự lạ
(dense_only > no_rerank > full) là hệ quả toán học của định nghĩa single-gold,
không phải máy hỏng.

## Hố sâu nhất, mổ hết bằng thực nghiệm (Bảng 4)

<img src="report/figs/fig_faith_gap.png" width="650">

Faithfulness 0,41 so với 0,89, chẻ thành 3 lớp đã đo: câu ngắn của ta khớp
paper (0,89 — máy nguyên vẹn); chấm đúng 2 bước trên cùng dữ liệu lên 0,77
(~75% hố do chấm tắt 1 bước); còn lại là judge khó hơn + dữ liệu nhiễu hơn
(không có GPT-5.5 — ghi rõ giới hạn). Cố tình không "sửa" bằng cách bóp cách
chấm: đó là gian lận chỉ số.

## Thứ hạng giữ nguyên cả 2 prompt (Bảng 5)

<img src="report/figs/fig_tab5.png" width="750">

ADGR 0,858 ≈ Mix 0,858 > Hybrid 0,829 > standalone, trung bình 5 lượt chấm độc
lập (SEM 0,046, ngắn thứ hai). Dưới prompt gốc của tác giả mọi mô hình thấp
hơn ~0,08 (ADGR 0,778) nhưng **0 đảo hạng trên 10 phương pháp** — kết luận so
sánh đứng vững bất kể thang chấm.

## Gọn mà trúng: case Aescin (Bảng 8)

<img src="report/figs/fig_tab8_case.png" width="700">

"Aescin chống chỉ định với ai?" Hybrid lôi 139 relation — trong đó có chống
chỉ định của thuốc khác (đúng khái niệm, sai entity, dễ gây hại). ADGR chỉ giữ
3 relation cần của Aescin. Ta: 12–14 entity / 39–41 relation — cùng phong cách
gọn. Trục log là bắt buộc ở đây: 139 so với 5 chênh 28 lần.

## Kết quả so với paper

| Bảng | Chỉ số | Paper | Ta | Kết luận |
|---|---|---|---|---|
| 1 | chunk / entity / relation | 17236 / 49274 / 106666 | 15837 / 52578 / 106279 | KHỚP (92% / −0,4%) |
| 1 | DSER / SCR / LCR | 0.936 / 0.928 / 0.906 | 0.913 / 0.884 / 0.816 | GẦN–GIẢI THÍCH ĐƯỢC |
| 3 | Hit@1 / Hit@10 bản đủ | 0.5967 / 0.7100 | 0.4333 / 0.7933 ; 0.6833* | KHỚP (trong khoảng proxy) |
| 3 | MRR / nDCG@10 bản đủ | 0.6437 / 0.6606 | 0.5538* / 0.6118* | GẦN |
| 4 | RAGAS / faithfulness | 0.8792 / 0.8878 | 0.64 / 0.41 (0.77 hai bước) | GIẢI THÍCH 3 lớp |
| 5 | ADGR tổng | 0.7921 | 0.858 / 0.778 (prompt gốc) | KHỚP, giữ hạng |
| 6 | token vào/dòng; tìm/sinh | 4490.5; 3.19s / 6.66s | 3441.4; 2.36s / 5.82s | VƯỢT (−23%, 9.85s→8.18s) |
| 7 | 5 arm (RAGAS paper) | 0.60–0.88 | chênh đã rõ | KHỚP hướng |
| 8 | Aescin H/M/ADGR | 7/139, 7/80, 5/57 | 12–14 / 39–41 | KHỚP phong cách |

\*proxy tự động (paper dùng gold labels chấm tay). Bảng verdict đầy đủ kèm
nguyên nhân một dòng nằm trong báo cáo.

## Thay thế đã công khai (đo hết, không giấu cái nào)

| Paper | Ta | Ảnh hưởng đã đo |
|---|---|---|
| Mô hình sinh thương mại | Qwen2.5-14B-Instruct-AWQ, temp 0, vLLM local | đáp án quyết định luận |
| Embedding thương mại | embeddinggemma:300m (768 chiều) | index dựng lại −0,4% relation |
| Reranker thương mại | Qwen3-Reranker-0.6B (+ fallback CPU) | đủ score đã verify |
| Judge GPT-5.5 | solar-pro4, cùng protocol | lệch thang ~0,08, giữ hạng |
| 7 standalone trả phí | 7 mô hình miễn phí (khác hẳn) | chỉ tham khảo |
| Gold labels chấm tay | slug-gold + single-gold (đồng thuận) | số paper nằm trong khoảng |
| Chunker tác giả | Suy ngược quy tắc H2 (91,9%) | 3,3 ent/chunk, LCR −9 điểm |
| Lib RAGAS gốc | Cài lại đúng công thức gốc | ans.rel 0.66 / rec 0.64 / prec 0.86 |
| Faithfulness 2 bước | 1 bước (rẻ) + kiểm tra 2 bước | 0.41 so với 0.77 cùng dữ liệu |

## Trong repo có gì

```
├── README.md / README.vi.md  # file này, Anh + Việt
├── AGENTS.md                  # kỷ luật vận hành đúc từ 43 lỗi đã sửa
├── LICENSE                    # MIT
├── report/                    # báo cáo học thuật (Anh + Việt, LaTeX + PDF) + 10 hình
├── src/                       # pipeline: tìm kiếm, đánh giá, chain 16 stages
├── results/                   # FINAL_v2.json (8 bảng) + toàn bộ đáp án/chấm thô
└── corpus/                    # bài báo, chunk, trích xuất, ngữ cảnh (<100MB mỗi file)
```

File lớn để ngoài GitHub đúng chuẩn (hướng dẫn dựng lại trong `AGENTS.md`):
index tác giả (2.3GB), index tự dựng (960MB), models (9.4GB).

## Phương pháp vắn tắt

Câu hỏi → keyword LLM (memoize, 1 gọi/câu) → anchor lai (BM25 top-10 ∪ dense
top-10, thưởng chồng lấp, top-15) → edge 1 bước, lọc edge, quota m, trần Nmax
→ lưới an toàn global + direct → merge (⌊rn/2⌋ mỗi nguồn, round-robin) →
rerank top-10 → Qwen2.5-14B sinh đáp án (temp 0) → chấm Hit/MRR/nDCG, RAGAS
đúng công thức gốc, judge 4 chiều × 5 lượt (leo thang minimal×3 → low×1, rẻ
28 lần; quota guard dừng ầm ĩ).

## Chạy lại

```bash
python -u src/kg_build_paper.py   # dựng index đồ thị
python -u src/chain_full.py       # chạy toàn bộ (lưu từng câu, resume được,
                                  # tự kiểm tra mỗi stage)
```

Cần: 1 GPU 24GB, vLLM + ollama, Python 3.11, key Upstage (~$5 cho toàn bộ
protocol chấm). Đọc `AGENTS.md` trước — mỗi quy tắc ở đó tồn tại vì vi phạm nó
từng tốn hàng giờ GPU hoặc làm hỏng cả bảng số.

## Trích dẫn paper gốc

```bibtex
@article{nguyentienha2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based
             Retrieval-Augmented Generation},
  author  = {{Nguyen Tien Ha} and {Vu Duy Linh} and {Nguyen Thi Minh Huyen}
             and {Le Ngoc Toan} and {Kim Dinh Thai}},
  journal = {Applied Intelligence (submitted)},
  year    = {2026}
}
```
