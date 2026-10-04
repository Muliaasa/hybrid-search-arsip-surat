import argparse, os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_ID = "firqaaa/indo-sentence-bert-base"
SEED = 42

SYSTEM_DOC_IDS = [6, 16, 17, 19, 27, 38, 39, 40, 49, 66, 67, 68, 70, 78, 79, 86, 87, 88, 90, 99, 122]

FALLBACK_STOPWORDS = set("""yang dan di ke dari untuk dengan pada guna agar dalam atau oleh bagi serta sebagai ini itu
akan telah dapat tersebut antar para atas terhadap sesuai melalui kepada karena saat secara seluruh perlu
adalah juga lebih agar sehingga yaitu mengenai sebuah setiap dalam""".split())

# ----------------------------------------------------------------- util teks
def load_stopwords():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        sw = set(StopWordRemoverFactory().get_stop_words())
        return sw | FALLBACK_STOPWORDS
    except Exception:
        print("[info] Sastrawi tidak terpasang; memakai daftar kata henti cadangan.", file=sys.stderr)
        return FALLBACK_STOPWORDS


def tokenize(text, stop):
    toks = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in toks if t not in stop and len(t) > 1]


# ------------------------------------------------------------------ peringkat
def rank_desc(scores):
    """indeks dokumen terurut menurun (stabil)."""
    return np.argsort(-scores, kind="stable")


def rrf(rank_lists, k=60, tiebreak=None, n_docs=None):
    """Reciprocal Rank Fusion atas daftar peringkat penuh. rank mulai dari 1.
    Seri diurutkan menurut skor tiebreak (mis. skor semantik) bila tersedia."""
    n = n_docs if n_docs is not None else len(rank_lists[0])
    s = np.zeros(n)
    for lst in rank_lists:
        for r, d in enumerate(lst, start=1):
            s[d] += 1.0 / (k + r)
    tb = tiebreak if tiebreak is not None else -np.arange(n, dtype=float)
    order = np.lexsort((-tb, -s))         
    return order, s


# --------------------------------------------------------------------- metrik
def precision_at(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else np.nan


def average_precision(ranked, rel):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel) if rel else np.nan


def ndcg_at(ranked, rel, k):
    dcg = sum(1.0 / np.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in rel)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else np.nan


def mrr(ranked, rel):
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def all_metrics(ranked, rel):
    return dict(P5=precision_at(ranked, rel, 5), R10=recall_at(ranked, rel, 10),
                AP=average_precision(ranked, rel), nDCG10=ndcg_at(ranked, rel, 10), MRR=mrr(ranked, rel))


# ---------------------------------------------------------------- anotasi/qrels
def load_qrels(args, queries, ids_in_corpus):
    qrels = {}
    if os.path.exists(args.anotasi):
        with open(args.anotasi, encoding="utf-8-sig", errors="replace") as fh:
            head = fh.readline()
        sep = ";" if head.count(";") > head.count(",") else ","      # Excel berlokal Indonesia sering memakai titik koma
        an = pd.read_csv(args.anotasi, sep=sep, encoding="utf-8-sig")
        an.columns = [str(c).strip().lower() for c in an.columns]

        def _kappa(x, y):
            po = (x == y).mean(); p1, p2 = x.mean(), y.mean()
            pe = p1 * p2 + (1 - p1) * (1 - p2)
            return po, ((po - pe) / (1 - pe) if pe < 1 else np.nan)

        a1 = pd.to_numeric(an["a1"], errors="coerce")
        if a1.isna().any():
            sys.exit(f"Kolom a1 harus terisi 0/1 pada semua baris (kosong: {int(a1.isna().sum())}).")
        a2 = pd.to_numeric(an["a2"], errors="coerce") if "a2" in an.columns else pd.Series(np.nan, index=an.index)
        fin = pd.to_numeric(an["final"], errors="coerce") if "final" in an.columns else pd.Series(np.nan, index=an.index)
        both = a1.notna() & a2.notna()
        lab = a1.copy()
        if both.sum() > 0:
            po, kp = _kappa(a1[both], a2[both])
            print(f"[anotasi] anotator 1 & 2 dibandingkan pada {int(both.sum())} dari {len(an)} pasangan: "
                  f"kesepakatan={po:.3f}  Cohen's kappa={kp:.3f}")
            diff = both & (a1 != a2)
            if diff.any():
                if fin[diff].isna().any():
                    sys.exit(f"{int(diff.sum())} pasangan berselisih, tetapi kolom 'final' belum terisi pada sebagian baris.")
                lab[diff] = fin[diff]
        else:
            print(f"[anotasi] satu anotator ({len(an)} pasangan); Cohen's kappa antaranotator tidak dihitung.")
        if "a1_ulang" in an.columns:
            r = pd.to_numeric(an["a1_ulang"], errors="coerce"); m = r.notna()
            if m.sum() > 0:
                po, kp = _kappa(a1[m], r[m])
                print(f"[anotasi] konsistensi penilaian ulang oleh anotator yang sama: {int(m.sum())} pasangan, "
                      f"kesepakatan={po:.3f}  kappa intra-anotator={kp:.3f}")
        an["lab"] = lab.astype(int)
        for q, g in an.groupby("qid"):
            qrels[q] = set(g.loc[g.lab == 1, "doc_id"].astype(int))
    elif args.use_candidate:
        print("[PERINGATAN] memakai id_relevan_kandidat sebagai ground truth. Hanya untuk uji coba kode; "
              "JANGAN dilaporkan di artikel sebelum dinilai anotator independen.", file=sys.stderr)
        for r in queries.itertuples():
            qrels[r.qid] = {int(x) for x in str(r.id_relevan_kandidat).split(";") if x.strip()}
    else:
        sys.exit("anotasi.csv tidak ditemukan. Jalankan dahulu tanpa metrik untuk membuat pool "
                 "(--pool_only), lalu isi anotasi.csv, atau gunakan --use_candidate hanya untuk uji coba.")
    return {q: {d for d in rel if d in ids_in_corpus} for q, rel in qrels.items()}


# ------------------------------------------------------------------ klaster
def division_code(nomor):
    m = re.search(r"-([A-Z]+)\.[A-Z0-9]+/", nomor)
    return m.group(1) if m else None


def ctfidf_keywords(texts, labels, stop, topn=5):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(tokenizer=lambda t: tokenize(t, stop), lowercase=False, token_pattern=None)
    labs = sorted(set(labels) - {-1})
    if not labs:
        return {}
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in labs]
    X = cv.fit_transform(docs).toarray().astype(float)
    tf = X / X.sum(axis=1, keepdims=True)
    A = X.sum() / len(labs)
    idf = np.log(1 + A / X.sum(axis=0))
    W = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {c: list(vocab[np.argsort(-W[i])[:topn]]) for i, c in enumerate(labs)}


def run_clusters(df, emb, stop, outdir, args):
    import umap, hdbscan
    from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score, adjusted_rand_score
    texts = (df.perihal + " " + df.ocr).tolist()
    div = df.nomor.map(division_code)
    blokA = (df.id <= 100).values

    def fit(n_comp, mcs, ms=None, md=0.0, seed=SEED):
        red = umap.UMAP(n_neighbors=15, n_components=n_comp, min_dist=md, metric="cosine",
                        random_state=seed).fit_transform(emb)
        lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=False).fit_predict(red)
        return red, lab

    def summarize(red, lab, n_comp, mcs, md=0.0, seed=SEED):
        nz = lab != -1
        k = len(set(lab[nz]))
        sil_umap = silhouette_score(red[nz], lab[nz]) if k > 1 else np.nan
        sil_orig = silhouette_score(emb[nz], lab[nz], metric="cosine") if k > 1 else np.nan
        yA = div[blokA].values
        nmi_all = normalized_mutual_info_score(yA, lab[blokA]); ari_all = adjusted_rand_score(yA, lab[blokA])
        m = blokA & nz
        nmi_nn = normalized_mutual_info_score(div[m].values, lab[m]) if m.sum() > 1 else np.nan
        ari_nn = adjusted_rand_score(div[m].values, lab[m]) if m.sum() > 1 else np.nan
        return dict(n_components=n_comp, min_cluster_size=mcs, min_dist=md, seed=seed, n_klaster=k, n_noise=int((~nz).sum()),
                    persen_noise=round(100 * (~nz).mean(), 1), sil_ruang_umap=sil_umap, sil_ruang_asli=sil_orig,
                    NMI_dgn_noise=nmi_all, ARI_dgn_noise=ari_all, NMI_tanpa_noise=nmi_nn, ARI_tanpa_noise=ari_nn)

    red, lab = fit(2, 15)
    main = summarize(red, lab, 2, 15)
    print("\n[klaster utama]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in main.items()})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = plt.get_cmap("tab10")
        grp = div.where(blokA, "Blok B").fillna("Blok B")
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        for t in sorted(set(lab)):
            m = lab == t
            axs[0].scatter(red[m, 0], red[m, 1], s=30, alpha=0.85, edgecolors="none",
                           color="#b5b5b5" if t == -1 else cm(t % 10),
                           label="Noise (-1)" if t == -1 else f"Topik {t}")
        axs[0].set_title(f"(a) Klaster HDBSCAN (min_cluster_size={15})", fontsize=10)
        pal = {"PM": "#1f77b4", "SDM": "#ff7f0e", "KEU": "#2ca02c", "IT": "#d62728", "Blok B": "#9e9e9e"}
        for g_, c_ in pal.items():
            m = (grp == g_).values
            axs[1].scatter(red[m, 0], red[m, 1], s=30, alpha=0.85, edgecolors="none", color=c_, label=g_)
        ms = df.id.isin(SYSTEM_DOC_IDS).values
        axs[1].scatter(red[ms, 0], red[ms, 1], s=70, facecolors="none", edgecolors="black", linewidths=1.0,
                       label="Tema sistem (21)")
        axs[1].set_title("(b) Divisi penerbit (Blok A), Blok B, dan dokumen bertema sistem", fontsize=10)
        for a_ in axs:
            a_.set_xlabel("UMAP dimensi 1"); a_.set_ylabel("UMAP dimensi 2"); a_.grid(alpha=0.25)
            a_.legend(fontsize=7, loc="best", framealpha=0.9)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "gambar_umap.png"), dpi=220)
        plt.close(fig)
    except Exception as e:
        print("[info] gambar UMAP tidak dibuat:", e, file=sys.stderr)
    df_out = df[["id", "nomor", "perihal"]].copy()
    df_out["kode_divisi"] = div; df_out["blok"] = np.where(blokA, "A (1-100)", "B (101-150)")
    df_out["topik"] = lab
    df_out["bertema_sistem"] = df.id.isin(SYSTEM_DOC_IDS)
    df_out.to_csv(os.path.join(outdir, "klaster_penugasan.csv"), index=False)
    # silhouette per klaster (ruang UMAP)
    from sklearn.metrics import silhouette_samples
    nz = lab != -1
    if len(set(lab[nz])) > 1:
        ss = silhouette_samples(red[nz], lab[nz])
        per = pd.Series(ss).groupby(lab[nz]).mean().round(4)
        print("[silhouette per klaster, ruang UMAP]\n", per.to_string())
    kw = ctfidf_keywords(texts, lab, stop)
    pd.DataFrame([{"topik": c, "n_dokumen": int((lab == c).sum()), "kata_kunci": ", ".join(w)}
                  for c, w in kw.items()]).to_csv(os.path.join(outdir, "klaster_kata_kunci.csv"), index=False)
    # sebaran per blok dan per divisi
    print("\n[noise per blok]\n", pd.crosstab(df_out.blok, df_out.topik).to_string())
    print("\n[blok A: divisi x topik]\n", pd.crosstab(df_out.loc[blokA, "kode_divisi"], df_out.loc[blokA, "topik"]).to_string())
    print("\n[dokumen bertema sistem x topik]\n", pd.crosstab(df_out.bertema_sistem, df_out.topik).to_string())
    pd.crosstab(df_out.loc[blokA, "kode_divisi"], df_out.loc[blokA, "topik"]).to_csv(os.path.join(outdir, "klaster_divisi_x_topik.csv"))
    pd.crosstab(df_out.blok, df_out.topik).to_csv(os.path.join(outdir, "klaster_blok_x_topik.csv"))
    rows = []
    for md in (0.0, 0.1):
        for nc in (2, 5):
            for mcs in (5, 8, 10, 15):
                r_, l_ = fit(nc, mcs, md=md); rows.append(summarize(r_, l_, nc, mcs, md))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(outdir, "klaster_sensitivitas.csv"), index=False)
    print("\n[sensitivitas: n_components x min_cluster_size x min_dist]\n", pd.DataFrame(rows).round(3).to_string(index=False))
    seeds = []
    for sd in (42, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        r_, l_ = fit(2, 15, seed=sd); seeds.append(summarize(r_, l_, 2, 15, 0.0, sd))
    sd_df = pd.DataFrame(seeds); sd_df.round(4).to_csv(os.path.join(outdir, "klaster_seed.csv"), index=False)
    cols_ = ["n_klaster", "persen_noise", "sil_ruang_umap", "sil_ruang_asli", "NMI_dgn_noise", "ARI_dgn_noise"]
    print("\n[stabilitas 10 seed, konfigurasi utama: rerata (simpangan baku)]")
    for c in cols_:
        print(f"  {c:16s} {sd_df[c].astype(float).mean():.3f} ({sd_df[c].astype(float).std():.3f})  rentang {sd_df[c].astype(float).min():.3f}-{sd_df[c].astype(float).max():.3f}")

# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_bersih.csv")
    ap.add_argument("--queries", default="kueri_relevansi.csv")
    ap.add_argument("--anotasi", default="anotasi.csv")
    ap.add_argument("--outdir", default="hasil")
    ap.add_argument("--semantic", action="store_true", help="hitung embedding IndoSBERT (butuh internet/HF)")
    ap.add_argument("--fake_semantic", action="store_true", help="HANYA UJI KODE: pakai SVD-TF-IDF sbg pengganti embedding")
    ap.add_argument("--clusters", action="store_true")
    ap.add_argument("--only_clusters", action="store_true", help="hanya analisis klaster (tidak butuh kueri/anotasi)")
    ap.add_argument("--exclude_system", action="store_true", help="ablasi: buang 21 dokumen bertema sistem e-arsip")
    ap.add_argument("--use_candidate", action="store_true", help="HANYA UJI KODE: pakai kandidat sebagai ground truth")
    ap.add_argument("--pool_only", action="store_true")
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--show", default=None, help="cetak 5 besar tiap sistem untuk satu kueri (bahan studi kasus)")
    ap.add_argument("--show_ids", default="", help="id dokumen (dipisah koma) yang peringkat/skornya ingin dicetak pada --show")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    need = [args.data] if args.only_clusters else [args.data, args.queries]
    missing = [f for f in need if not os.path.exists(f)]
    if missing:
        sys.exit("Berkas tidak ditemukan: %s\nBerkas di folder ini: %s\n"
                 "Unggah berkas tsb ke panel kiri Colab (nama harus persis, tanpa ' (1)')." % (missing, sorted(os.listdir("."))))

    df = pd.read_csv(args.data)
    if args.exclude_system:
        df = df[~df.id.isin(SYSTEM_DOC_IDS)].reset_index(drop=True)
        print(f"[ablasi] {len(SYSTEM_DOC_IDS)} dokumen bertema sistem dibuang; sisa {len(df)} dokumen")
    ids = df.id.values
    id2idx = {d: i for i, d in enumerate(ids)}
    raw_text = (df.perihal + ". " + df.ocr).tolist()
    stop = load_stopwords()
    toks = [tokenize(t, stop) for t in raw_text]

    # ---- indeks leksikal
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tfv = TfidfVectorizer(analyzer=lambda x: x, norm="l2")      # input = token terproses; vektor L2 => cosine
    Xtf = tfv.fit_transform(toks)
    print(f"[TF-IDF] matriks {Xtf.shape}, kepadatan {Xtf.nnz / (Xtf.shape[0] * Xtf.shape[1]):.4f}")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(toks)                                       # k1=1.5, b=0.75 (bawaan)

    emb = None
    if args.semantic:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_ID)
        emb = model.encode(raw_text, normalize_embeddings=True, show_progress_bar=False)  # teks mentah, tanpa buang kata henti
        np.save(os.path.join(args.outdir, "embeddings.npy"), emb)
    elif args.fake_semantic:
        from sklearn.decomposition import TruncatedSVD
        emb = TruncatedSVD(64, random_state=SEED).fit_transform(Xtf)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        print("[PERINGATAN] --fake_semantic aktif: hasil TIDAK boleh dilaporkan.", file=sys.stderr)
    have_sem = emb is not None

    if args.only_clusters:
        if not have_sem:
            sys.exit("--only_clusters membutuhkan --semantic")
        run_clusters(df, emb, stop, args.outdir, args)
        return

    queries = pd.read_csv(args.queries)

    def lexical_scores(q):
        qt = tokenize(q, stop)
        s_tf = cosine_similarity(tfv.transform([qt]), Xtf).ravel()
        s_bm = np.asarray(bm25.get_scores(qt))
        return s_tf, s_bm

    def sem_scores(q):
        qe = model.encode([q], normalize_embeddings=True)[0] if args.semantic else None
        if args.fake_semantic:
            from sklearn.decomposition import TruncatedSVD
            qv = tfv.transform([tokenize(q, stop)]); qe = (qv @ svd_comp.T).ravel()
            qe = qe / (np.linalg.norm(qe) + 1e-9)
        return emb @ qe

    if args.fake_semantic:
        from sklearn.decomposition import TruncatedSVD
        _svd = TruncatedSVD(64, random_state=SEED).fit(Xtf); svd_comp = _svd.components_

    def run_systems(q, k=args.rrf_k):
        s_tf, s_bm = lexical_scores(q)
        rk = {"TF-IDF": rank_desc(s_tf), "BM25": rank_desc(s_bm)}
        if have_sem:
            s_se = sem_scores(q)
            rk["IndoSBERT"] = rank_desc(s_se)
            rk["RRF(TF-IDF+IndoSBERT)"] = rrf([rk["TF-IDF"], rk["IndoSBERT"]], k, s_se, len(df))[0]
            rk["RRF(BM25+IndoSBERT)"] = rrf([rk["BM25"], rk["IndoSBERT"]], k, s_se, len(df))[0]
        return rk

    if args.show:
        s_tf, s_bm = lexical_scores(args.show)
        print(f"\n[studi kasus] kueri: {args.show!r}")
        def top5(name, sc):
            print(f"-- {name}")
            for r_, i in enumerate(rank_desc(sc)[:5], 1):
                print(f"   {r_}. #{ids[i]:<4} {sc[i]:.4f}  {df.perihal[i]}")
        top5("TF-IDF", s_tf); top5("BM25", s_bm)
        if have_sem:
            s_se = sem_scores(args.show); top5("IndoSBERT", s_se)
            sem_pos = {int(i): r for r, i in enumerate(rank_desc(s_se), 1)}
            for nm, lst in (("RRF(TF-IDF+IndoSBERT)", rank_desc(s_tf)), ("RRF(BM25+IndoSBERT)", rank_desc(s_bm))):
                order, sc = rrf([lst, rank_desc(s_se)], args.rrf_k, s_se, len(df))
                lex_pos = {int(i): r for r, i in enumerate(lst, 1)}
                print(f"-- {nm}   [peringkat jalur: leksikal / semantik]")
                for r_, i in enumerate(order[:5], 1):
                    print(f"   {r_}. #{ids[i]:<4} {sc[i]:.5f}  {df.perihal[i]}  [{lex_pos[int(i)]} / {sem_pos[int(i)]}]")
                if args.show_ids:
                    rrf_pos = {int(i): r for r, i in enumerate(order, 1)}
                    for d_ in [int(x) for x in args.show_ids.split(",") if x.strip()]:
                        if d_ in id2idx:
                            i = id2idx[d_]
                            print(f"   (#{d_}) skor RRF {sc[i]:.5f}, peringkat RRF {rrf_pos[i]}, leksikal {lex_pos[i]}, semantik {sem_pos[i]}  {df.perihal[i]}")

    pool = []
    for r in queries.itertuples():
        rk = run_systems(r.kueri)
        cand = {int(x) for x in str(r.id_relevan_kandidat).split(";") if x.strip()}
        docs = set()
        for lst in rk.values():
            docs |= {int(ids[i]) for i in lst[:10]}
        docs |= {d for d in cand if d in id2idx}
        for d in sorted(docs):
            pool.append(dict(qid=r.qid, kueri=r.kueri, doc_id=d, perihal=df.perihal[id2idx[d]],
                             isi=df.ocr[id2idx[d]], a1="", a2="", final=""))
    pd.DataFrame(pool).to_csv(os.path.join(args.outdir, "pool_untuk_anotasi.csv"), index=False)
    print(f"[pool] {len(pool)} pasangan (kueri, dokumen) disimpan ke pool_untuk_anotasi.csv")
    if args.pool_only:
        return

    # ---- evaluasi
    qrels = load_qrels(args, queries, set(ids))
    rows = []
    for r in queries.itertuples():
        rel_idx = {id2idx[d] for d in qrels.get(r.qid, set())}
        if not rel_idx:
            continue
        for name, lst in run_systems(r.kueri).items():
            rows.append(dict(qid=r.qid, tipe=r.tipe, bidang=r.bidang, sistem=name, **all_metrics(list(lst), rel_idx)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.outdir, "metrik_per_kueri.csv"), index=False)
    cols = ["P5", "R10", "AP", "nDCG10", "MRR"]
    tab = res.groupby("sistem")[cols].mean().round(4)
    tab.rename(columns={"AP": "MAP"}).to_csv(os.path.join(args.outdir, "tabel_metrik.csv"))
    print("\n[rata-rata metrik, N kueri =", res.qid.nunique(), "]\n", tab.rename(columns={"AP": "MAP"}).to_string())
    for tipe, g in res.groupby("tipe"):
        print(f"\n[tipe kueri {tipe}]\n", g.groupby("sistem")[cols].mean().round(4).rename(columns={"AP": "MAP"}).to_string())

    if have_sem:
        from scipy.stats import wilcoxon
        base = "RRF(TF-IDF+IndoSBERT)"
        out = []
        piv = {m: res.pivot(index="qid", columns="sistem", values=m) for m in ("nDCG10", "AP")}
        for m, P in piv.items():
            for other in P.columns:
                if other == base:
                    continue
                d = P[base] - P[other]
                p = np.nan if (d == 0).all() else wilcoxon(P[base], P[other], zero_method="wilcox").pvalue
                out.append(dict(metrik=m, pembanding=other, selisih_rata2=round(d.mean(), 4), p_value=p))
        w = pd.DataFrame(out)
        w["p_holm"] = np.nan
        for m in w.metrik.unique():
            idx = w.index[w.metrik == m]
            ps = w.loc[idx, "p_value"].values
            order = np.argsort(ps); adj = np.empty_like(ps, dtype=float); mx = 0
            for rank, i in enumerate(order):
                mx = max(mx, (len(ps) - rank) * ps[i]); adj[i] = min(1, mx)
            w.loc[idx, "p_holm"] = adj
        w.round(4).to_csv(os.path.join(args.outdir, "uji_wilcoxon.csv"), index=False)
        print("\n[Wilcoxon, referensi = RRF(TF-IDF+IndoSBERT)]\n", w.round(4).to_string(index=False))

        rows = []
        for k in (10, 30, 60, 100):
            for r in queries.itertuples():
                rel_idx = {id2idx[d] for d in qrels.get(r.qid, set())}
                if not rel_idx:
                    continue
                rk = run_systems(r.kueri, k)["RRF(TF-IDF+IndoSBERT)"]
                rows.append(dict(k=k, **all_metrics(list(rk), rel_idx)))
        ab = pd.DataFrame(rows).groupby("k")[cols].mean().round(4).rename(columns={"AP": "MAP"})
        ab.to_csv(os.path.join(args.outdir, "ablasi_k_rrf.csv"))
        print("\n[ablasi k pada RRF(TF-IDF+IndoSBERT)]\n", ab.to_string())

    if args.clusters:
        if not have_sem:
            print("--clusters membutuhkan --semantic", file=sys.stderr)
        else:
            run_clusters(df, emb, stop, args.outdir, args)


if __name__ == "__main__":
    main()
