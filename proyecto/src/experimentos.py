"""Pasos 3 a 7 de la diapositiva 4: división, representación, línea base,
ajuste de hiperparámetros y evaluación final.

Uso:  python proyecto/src/depurar.py   (una vez)
      python proyecto/src/experimentos.py
Salidas:
  proyecto/resultados/*.csv|json      métricas agregadas (sin texto de comentarios)
  proyecto/data/processed/errores_test.csv   errores con texto (no se sube a git)
  proyecto/modelos/modelo_polaridad.joblib + metadata.json
  paper/figuras/*.png
"""
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocesamiento import normalizar_basico, normalizar_marcado  # noqa: E402
from representaciones import EmbeddingsFrase  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
CLASES = ["negativo", "neutral", "positivo"]
RAIZ = Path(__file__).resolve().parent.parent
REPO = RAIZ.parent
DATOS = RAIZ / "data" / "processed" / "dataset_depurado.csv"
OUT = RAIZ / "resultados"
MOD = RAIZ / "modelos"
FIG = REPO / "paper" / "figuras"
for p in (OUT, MOD, FIG):
    p.mkdir(parents=True, exist_ok=True)

# Paleta del Informe_ML
AZUL, NARANJA, GRIS = "#1b2a6b", "#f2a51a", "#8a94a6"
COLORES_CLASE = {"negativo": "#c0392b", "neutral": GRIS, "positivo": "#2e86c1"}
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150})

EMB_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NOMBRE_REP = {"tfidf_pal": "TF-IDF palabras", "tfidf_car": "TF-IDF caracteres",
              "emb_minilm": "Emb. MiniLM", "emb_mpnet": "Emb. MPNet"}
NOMBRE_CLF = {"dummy_freq": "Dummy (mayoritaria)", "dummy_estrat": "Dummy (estratificado)",
              "lr": "Regresión logística", "svm": "SVM lineal", "nb": "Naive Bayes",
              "rf": "Random Forest"}
resumen = {}

# ---------------------------------------------------------------- datos
df = pd.read_csv(DATOS, encoding="utf-8-sig")
df = df[df["feedback"].fillna("").str.strip() != ""].reset_index(drop=True)
X, y = df["feedback"], df["polaridad_basica"]
resumen["n_total"] = len(df)
resumen["distribucion"] = y.value_counts().reindex(CLASES).to_dict()
resumen["sha256_dataset"] = hashlib.sha256(DATOS.read_bytes()).hexdigest()[:16]

# ------------------------------------------------- paso 3: grupos y división
# No hay columna de autor: se aproxima con grupos de comentarios casi idénticos
# (similitud coseno TF-IDF >= 0.7) para que nunca queden repartidos entre train y test.
S = cosine_similarity(TfidfVectorizer().fit_transform(X))
np.fill_diagonal(S, 0)
n_grupos, grupos = connected_components(S >= 0.7, directed=False)
resumen["n_grupos"] = int(n_grupos)
resumen["pares_casi_duplicados"] = int((np.triu(S) >= 0.7).sum())

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
idx_tr, idx_te = next(sgkf.split(X, y, grupos))
X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
g_tr = grupos[idx_tr]
resumen["n_train"], resumen["n_test"] = len(idx_tr), len(idx_te)
resumen["dist_train"] = y_tr.value_counts().reindex(CLASES).to_dict()
resumen["dist_test"] = y_te.value_counts().reindex(CLASES).to_dict()
assert not set(grupos[idx_tr]) & set(grupos[idx_te]), "fuga entre train y test"
cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_tr, y_tr, g_tr))
print(f"train={len(idx_tr)} test={len(idx_te)} grupos={n_grupos}")


# ------------------------------------------------- paso 4: representaciones
def representacion(nombre):
    if nombre == "tfidf_pal":
        return TfidfVectorizer(preprocessor=normalizar_marcado, ngram_range=(1, 2),
                               min_df=2, sublinear_tf=True)
    if nombre == "tfidf_car":
        return TfidfVectorizer(preprocessor=normalizar_basico, analyzer="char_wb",
                               ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    if nombre == "emb_minilm":
        return EmbeddingsFrase(EMB_MINILM)
    if nombre == "emb_mpnet":
        return EmbeddingsFrase(EMB_MPNET)


def clasificador(nombre, rep):
    es_emb = rep.startswith("emb")
    return {
        "dummy_freq": DummyClassifier(strategy="most_frequent"),
        "dummy_estrat": DummyClassifier(strategy="stratified", random_state=SEED),
        "lr": LogisticRegression(max_iter=5000, random_state=SEED),
        "svm": LinearSVC(random_state=SEED, max_iter=20000),
        "nb": GaussianNB() if es_emb else MultinomialNB(),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }[nombre]


def pipe(rep, clf):
    return Pipeline([("rep", representacion(rep)), ("clf", clasificador(clf, rep))])


REPS = ["tfidf_pal", "tfidf_car", "emb_minilm", "emb_mpnet"]
CLFS = ["lr", "svm", "nb", "rf"]
ESC = {"f1_macro": "f1_macro", "accuracy": "accuracy"}

# pre-cálculo de embeddings (cachea en memoria; el modelo no se ajusta con los datos)
t0 = time.time()
for m in (EMB_MINILM, EMB_MPNET):
    EmbeddingsFrase(m).transform(X)
resumen["seg_embeddings"] = round(time.time() - t0, 1)
print(f"embeddings listos en {resumen['seg_embeddings']} s")

# ------------------------------------------------- paso 5: línea base
filas = []
for clf in ["dummy_freq", "dummy_estrat"]:
    r = cross_validate(pipe("tfidf_pal", clf), X_tr, y_tr, cv=cv, scoring=ESC)
    filas.append({"rep": "-", "clf": clf, "f1_macro": r["test_f1_macro"].mean(),
                  "f1_std": r["test_f1_macro"].std(), "accuracy": r["test_accuracy"].mean()})
for rep in REPS:
    for clf in CLFS:
        r = cross_validate(pipe(rep, clf), X_tr, y_tr, cv=cv, scoring=ESC, n_jobs=1)
        filas.append({"rep": rep, "clf": clf, "f1_macro": r["test_f1_macro"].mean(),
                      "f1_std": r["test_f1_macro"].std(), "accuracy": r["test_accuracy"].mean()})
        print(f"base {rep:11s} {clf:4s} F1={filas[-1]['f1_macro']:.3f}")
cv_base = pd.DataFrame(filas)
cv_base.to_csv(OUT / "cv_linea_base.csv", index=False)

# ------------------------------------------------- paso 6: hiperparámetros
REJ_CLF = {
    "lr": {"clf__C": [0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]},
    "svm": {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
    "nb_tfidf": {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
    "nb_emb": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    "rf": {"clf__max_depth": [None, 20], "clf__min_samples_leaf": [1, 3],
           "clf__class_weight": [None, "balanced_subsample"]},
}
REJ_REP = {
    "tfidf_pal": {"rep__ngram_range": [(1, 1), (1, 2)], "rep__min_df": [1, 2],
                  "rep__preprocessor": [normalizar_basico, normalizar_marcado]},
    "tfidf_car": {"rep__ngram_range": [(2, 4), (3, 5)], "rep__min_df": [1, 2]},
    "emb_minilm": {}, "emb_mpnet": {},
}
filas, busquedas = [], {}
for rep in REPS:
    for clf in CLFS:
        clave = clf if clf != "nb" else ("nb_emb" if rep.startswith("emb") else "nb_tfidf")
        rejilla = {**REJ_REP[rep], **REJ_CLF[clave]}
        gs = GridSearchCV(pipe(rep, clf), rejilla, scoring="f1_macro", cv=cv,
                          n_jobs=-1 if not rep.startswith("emb") else 1, refit=True)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        params = {k: (v.__name__ if callable(v) else v) for k, v in gs.best_params_.items()}
        filas.append({"rep": rep, "clf": clf, "f1_macro": gs.best_score_,
                      "f1_std": gs.cv_results_["std_test_score"][i],
                      "n_combinaciones": len(gs.cv_results_["params"]),
                      "seg": round(time.time() - t0, 1), "mejores_params": json.dumps(params)})
        busquedas[(rep, clf)] = gs
        print(f"grid {rep:11s} {clf:4s} F1={gs.best_score_:.3f} {params}")
cv_aj = pd.DataFrame(filas).sort_values("f1_macro", ascending=False)
cv_aj.to_csv(OUT / "cv_ajustado.csv", index=False)

# efecto del preprocesamiento con marcado de negación/intensificadores (LR + TF-IDF palabras)
res = pd.DataFrame(busquedas[("tfidf_pal", "lr")].cv_results_)
res["prep"] = res["param_rep__preprocessor"].map(lambda f: f.__name__)
resumen["efecto_marcado"] = res.groupby("prep")["mean_test_score"].max().round(4).to_dict()

# ------------------------------------------------- paso 7: evaluación en test
mejor_rep, mejor_clf = cv_aj.iloc[0][["rep", "clf"]]
filas, preds = [], {}
for (rep, clf), gs in busquedas.items():
    p = gs.predict(X_te)
    preds[(rep, clf)] = p
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, p, average="macro", zero_division=0)
    filas.append({"rep": rep, "clf": clf, "accuracy": accuracy_score(y_te, p),
                  "precision_macro": pr, "recall_macro": rc, "f1_macro": f1})
for clf in ["dummy_freq", "dummy_estrat"]:
    p = pipe("tfidf_pal", clf).fit(X_tr, y_tr).predict(X_te)
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, p, average="macro", zero_division=0)
    filas.append({"rep": "-", "clf": clf, "accuracy": accuracy_score(y_te, p),
                  "precision_macro": pr, "recall_macro": rc, "f1_macro": f1})
test = pd.DataFrame(filas).sort_values("f1_macro", ascending=False)
test.to_csv(OUT / "test.csv", index=False)

final = busquedas[(mejor_rep, mejor_clf)].best_estimator_
p_final = preds[(mejor_rep, mejor_clf)]
pr, rc, f1, sop = precision_recall_fscore_support(y_te, p_final, labels=CLASES, zero_division=0)
por_clase = pd.DataFrame({"clase": CLASES, "precision": pr, "recall": rc, "f1": f1, "soporte": sop})
por_clase.to_csv(OUT / "test_por_clase.csv", index=False)
cm = confusion_matrix(y_te, p_final, labels=CLASES)

# intervalo de confianza bootstrap (95 %) del macro-F1 en test
rng = np.random.default_rng(SEED)
yt, yp = y_te.to_numpy(), np.asarray(p_final)
boot = [f1_score(yt[b], yp[b], average="macro")
        for b in (rng.integers(0, len(yt), len(yt)) for _ in range(2000))]
resumen["final"] = {
    "rep": mejor_rep, "clf": mejor_clf,
    "params": json.loads(cv_aj.iloc[0]["mejores_params"]),
    "cv_f1_macro": round(float(cv_aj.iloc[0]["f1_macro"]), 4),
    "test_accuracy": round(accuracy_score(y_te, p_final), 4),
    "test_f1_macro": round(f1_score(y_te, p_final, average="macro"), 4),
    "test_f1_ic95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
    "matriz_confusion": cm.tolist(),
}

# errores: análisis por conectores de contraste
contraste = X_te.str.lower().str.contains(r"\bpero\b|sin embargo|aunque|a pesar")
acierto = (yt == yp)
resumen["errores_por_contraste"] = {
    "con_contraste": {"n": int(contraste.sum()), "accuracy": round(float(acierto[contraste.to_numpy()].mean()), 4)},
    "sin_contraste": {"n": int((~contraste).sum()), "accuracy": round(float(acierto[~contraste.to_numpy()].mean()), 4)},
}
err = df.iloc[idx_te][["id_global", "curso", "feedback", "polaridad_basica"]].copy()
err["prediccion"] = yp
if hasattr(final, "predict_proba"):
    err["confianza"] = final.predict_proba(X_te).max(axis=1).round(3)
err[err["polaridad_basica"] != err["prediccion"]].to_csv(
    RAIZ / "data" / "processed" / "errores_test.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------- guardar modelo (para el MVP)
joblib.dump(final, MOD / "modelo_polaridad.joblib")
meta = {**resumen["final"], "clases": CLASES, "semilla": SEED, "fecha": time.strftime("%Y-%m-%d"),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "sha256_dataset": resumen["sha256_dataset"]}
(MOD / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------- figuras
# F1: distribución de clases y por semestre
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
cnt = y.value_counts().reindex(CLASES)
ax[0].bar(CLASES, cnt, color=[COLORES_CLASE[c] for c in CLASES])
for i, v in enumerate(cnt):
    ax[0].text(i, v + 4, f"{v} ({v / len(y):.0%})", ha="center", fontsize=9)
ax[0].set_title("Distribución de la polaridad"); ax[0].set_ylabel("comentarios")
ax[0].set_ylim(0, cnt.max() * 1.18)
ct = pd.crosstab(df["semestre"], y).reindex(columns=CLASES)
ct.plot(kind="bar", stacked=True, ax=ax[1], color=[COLORES_CLASE[c] for c in CLASES], rot=0, width=0.6)
ax[1].set_title("Polaridad por semestre"); ax[1].set_xlabel(""); ax[1].legend(title=None, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig1_distribucion.png", bbox_inches="tight"); plt.close()

# F2: longitud por clase
fig, ax = plt.subplots(figsize=(6.5, 3))
lon = X.str.split().str.len()
sns.boxplot(x=y, y=lon, order=CLASES, palette=COLORES_CLASE, ax=ax, width=0.5, hue=y, legend=False)
ax.set_xlabel(""); ax.set_ylabel("palabras por comentario"); ax.set_title("Longitud de los comentarios por clase")
plt.tight_layout(); plt.savefig(FIG / "fig2_longitud.png", bbox_inches="tight"); plt.close()

# F3: comparación CV (línea base vs ajustado)
m = cv_base[cv_base.rep != "-"].merge(cv_aj[["rep", "clf", "f1_macro", "f1_std"]], on=["rep", "clf"],
                                      suffixes=("_base", "_aj"))
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
for a, rep in zip(axs, REPS):
    s = m[m.rep == rep].set_index("clf").reindex(CLFS)
    xs = np.arange(len(CLFS))
    a.bar(xs - 0.2, s["f1_macro_base"], 0.38, color=GRIS, label="por defecto")
    a.bar(xs + 0.2, s["f1_macro_aj"], 0.38, yerr=s["f1_std_aj"], color=AZUL, label="ajustado",
          error_kw={"elinewidth": 0.8, "capsize": 2})
    a.axhline(cv_base.loc[cv_base.clf == "dummy_freq", "f1_macro"].iloc[0], color=NARANJA, ls="--", lw=1.2,
              label="Dummy (mayoritaria)")
    a.set_xticks(xs); a.set_xticklabels(["LR", "SVM", "NB", "RF"]); a.set_title(NOMBRE_REP[rep])
    a.set_ylim(0, 0.8)
axs[0].set_ylabel("macro-F1 (CV 5 pliegues)")
axs[0].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(FIG / "fig3_cv_comparacion.png", bbox_inches="tight"); plt.close()

# F4: matriz de confusión del modelo final
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASES, yticklabels=CLASES, ax=ax[0])
ax[0].set_title("Conteos"); ax[0].set_xlabel("predicción"); ax[0].set_ylabel("etiqueta real")
cmn = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=CLASES, yticklabels=CLASES,
            ax=ax[1], vmin=0, vmax=1)
ax[1].set_title("Normalizada por fila (recall)"); ax[1].set_xlabel("predicción"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "fig4_matriz_confusion.png", bbox_inches="tight"); plt.close()

# F5: términos más influyentes (LR + TF-IDF palabras ajustado)
lr_tf = busquedas[("tfidf_pal", "lr")].best_estimator_
voc = np.array(lr_tf.named_steps["rep"].get_feature_names_out())
coef = lr_tf.named_steps["clf"].coef_
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for a, k, c in zip(axs, range(3), CLASES):
    top = np.argsort(coef[k])[-10:]
    a.barh(voc[top], coef[k][top], color=COLORES_CLASE[c])
    a.set_title(f"Clase «{c}»"); a.tick_params(axis="y", labelsize=8)
axs[1].set_xlabel("coeficiente de la regresión logística")
plt.tight_layout(); plt.savefig(FIG / "fig5_terminos.png", bbox_inches="tight"); plt.close()

# F6: proyección t-SNE de los embeddings MPNet
E = EmbeddingsFrase(EMB_MPNET).transform(X)
Z = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca").fit_transform(
    PCA(n_components=50, random_state=SEED).fit_transform(E))
fig, ax = plt.subplots(figsize=(6, 4.2))
for c in CLASES:
    k = (y == c).to_numpy()
    ax.scatter(Z[k, 0], Z[k, 1], s=12, alpha=0.75, color=COLORES_CLASE[c], label=c)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
ax.set_title("Proyección t-SNE de los embeddings (MPNet)")
plt.tight_layout(); plt.savefig(FIG / "fig6_tsne.png", bbox_inches="tight"); plt.close()

(OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
print(cv_aj.to_string()); print(test.to_string()); print(por_clase.to_string())
