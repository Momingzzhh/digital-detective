# English manuscript

The file main.tex is the editable English journal-style manuscript; main.pdf is its compiled reading copy. The structure follows a general science/engineering journal pattern (title, abstract, keywords, numbered sections, figures and tables, discussion, conclusion, and numbered references). It is not the template of a specific journal. The author line should be replaced before submission.

The related formatting examples were the [CNKI-hosted paper template](https://cbck.cbpt.cnki.net/EditorAN/PromptPageInfo.aspx?c=3&t=v) and [science and technology academic writing guide](https://www.publishmedia.cbpt.cnki.net/portal/minio/webs/hcrh/media/web/2024/07/26/%E7%A7%91%E6%8A%80%E5%AD%A6%E6%9C%AF%E8%AE%BA%E6%96%87%E5%86%99%E4%BD%9C%E6%8C%87%E5%8D%97%E5%8F%8A%E6%A0%BC%E5%BC%8F%E5%8F%82%E8%80%83%EF%BC%88%E5%90%88%E6%88%90%E6%B6%A6%E6%BB%91%E6%9D%90%E6%96%99%EF%BC%89.pdf). Journal-specific requirements may differ.

The initial five-case local run and the separate 150-case experiment are reported independently. The 150-case aggregate is recorded in the follow-up report at ../data/follow_up_aggregate_report.json; the paper does not invent per-case predictions or training logs. The manuscript states the exact 112/150 = 74.7% whole-case result and the remaining audit materials needed for reproducibility.

Compile with Tectonic or another compatible LaTeX tool:

    tectonic main.tex
