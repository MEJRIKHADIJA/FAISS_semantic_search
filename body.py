import hashlib
import json
import os
import re
import sys
import unicodedata
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


if hasattr(sys.stdout, "buffer"):
    sys.stdout.reconfigure(encoding="utf-8")


CORPUS = [
    "Les prix à la consommation en Tunisie ont enregistré une légère progression de 0,3 % en mai 2026.",
    "La hausse est principalement portée par le renchérissement des produits alimentaires (+0,4 %).",
    "Les prix des viandes ont bondi de 4,2 % en mai.",
    "Une baisse a été enregistrée du côté des œufs (-7%), des fruits frais (-2,9 %) et des légumes (-1,3 %).",
    "Les tarifs des restaurants, cafés et hôtels ont augmenté de 0,6 %.",
    "Les services hôteliers ont augmenté de 3,5 % à l'approche de la saison estivale.",
    "Le secteur de l'habillement et des chaussures affiche une progression de 0,4 %.",
    "Les produits manufacturés contribuent à hauteur de 1,7 % à l'inflation globale.",
    "Les services contribuent à hauteur de 1,4 % à l'inflation globale.",
    "Les produits non alimentaires libres ont une contribution de 2,9 % à l'inflation.",
]

THEMES = {
    "1": "Quels produits alimentaires ont vu leurs prix baisser ?",
    "2": "Quel est l'impact du tourisme et des hôtels sur l'inflation ?",
    "3": "Quels secteurs contribuent le plus à l'inflation globale ?",
    "4": "Comment évoluent les prix de la viande en Tunisie ?",
    "5": "Quelle est la tendance générale de l'inflation en mai 2026 ?",
    "6": "Saisir ma propre requête",
}

MODEL_NAME = "all-MiniLM-L6-v2"
RESULT_COUNT = 3
FALLBACK_DIMENSION = 384

FRENCH_STOPWORDS = {
    "a",
    "au",
    "aux",
    "ce",
    "ces",
    "comment",
    "de",
    "des",
    "du",
    "en",
    "est",
    "et",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "ma",
    "ont",
    "pour",
    "qu",
    "que",
    "quel",
    "quelle",
    "quels",
    "sont",
    "sur",
    "un",
    "une",
    "vu",
}

EXPANSIONS = {
    "alimentaire": "alimentation alimentaires produits oeufs fruits legumes viandes",
    "alimentaires": "alimentation alimentaire produits oeufs fruits legumes viandes",
    "baisser": "baisse baissees diminution oeufs fruits legumes",
    "baisse": "baisser baissees diminution oeufs fruits legumes",
    "contribuent": "contribution contribuent produits manufactures services inflation globale",
    "contribution": "contribuent produits manufactures services inflation globale",
    "generale": "tendance progression hausse prix consommation inflation mai 2026",
    "globale": "inflation contribution produits manufactures services",
    "hotel": "hotels hoteliers restaurants cafes tourisme saison estivale services",
    "hotelier": "hotels hoteliers restaurants cafes tourisme saison estivale services",
    "hoteliers": "hotels hotelier restaurants cafes tourisme saison estivale services",
    "hotels": "hotel hoteliers restaurants cafes tourisme saison estivale services",
    "impact": "hausse contribution inflation services",
    "inflation": "prix consommation hausse progression contribution globale",
    "mai": "2026 consommation progression prix inflation",
    "prix": "consommation hausse baisse inflation produits services",
    "produits": "alimentaires manufactures non alimentaires prix inflation",
    "secteur": "secteurs produits manufactures services contribution inflation globale",
    "secteurs": "secteur produits manufactures services contribution inflation globale",
    "services": "hoteliers hotels restaurants cafes contribution inflation globale",
    "tendance": "generale progression hausse prix consommation inflation mai 2026",
    "tourisme": "hotels hoteliers restaurants cafes saison estivale services",
    "viande": "viandes prix hausse bondi",
    "viandes": "viande prix hausse bondi",
}


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in normalized if not unicodedata.combining(character))


class LocalTextEmbedder:
    def __init__(self, dimension=FALLBACK_DIMENSION):
        self.dimension = dimension

    def encode(self, texts, convert_to_numpy=True):
        vectors = np.vstack([self.encode_one(text) for text in texts]).astype("float32")
        if convert_to_numpy:
            return vectors
        return vectors.tolist()

    def encode_one(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature, weight in self.weighted_features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).hexdigest()
            position = int(digest, 16) % self.dimension
            vector[position] += weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def weighted_features(self, text):
        normalized = normalize_text(text)
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 1 and token not in FRENCH_STOPWORDS
        ]

        features = []
        for token in tokens:
            features.append((f"word:{token}", 1.0))
            for related_token in EXPANSIONS.get(token, "").split():
                features.append((f"word:{related_token}", 0.85))

        for first_token, second_token in zip(tokens, tokens[1:]):
            features.append((f"pair:{first_token}_{second_token}", 1.45))

        compact_text = " ".join(tokens)
        for size in (4, 5):
            for start in range(max(len(compact_text) - size + 1, 0)):
                features.append((f"chars:{compact_text[start:start + size]}", 0.12))

        return features or [("empty", 1.0)]


def load_embedding_model():
    print("Chargement du modèle sémantique...")
    if SentenceTransformer is not None:
        try:
            return SentenceTransformer(MODEL_NAME), "Sentence Transformers"
        except Exception as error:
            print(f"Impossible de charger {MODEL_NAME}: {error}")

    print("Mode hors ligne activé : moteur local de similarité.")
    return LocalTextEmbedder(), "Moteur local hors ligne"


model, backend_label = load_embedding_model()
embeddings = model.encode(CORPUS, convert_to_numpy=True).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"Index FAISS prêt ({backend_label}).")


HTML_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recherche sémantique — Inflation en Tunisie</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: rgba(255, 255, 255, 0.1);
      --card-strong: rgba(255, 255, 255, 0.16);
      --text: #f8fafc;
      --muted: #cbd5e1;
      --accent: #38bdf8;
      --accent-2: #a78bfa;
      --accent-3: #34d399;
      --shadow: rgba(2, 6, 23, 0.45);
      --danger: #fb7185;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient( rgba(56, 189, 248, 0.32), transparent 28rem),
        radial-gradient( rgba(167, 139, 250, 0.36), transparent 30rem),
        radial-gradient( rgba(52, 211, 153, 0.24), transparent 25rem),
        linear-gradient(145deg, #020617 0%, var(--bg) 55%, #111827 100%);
    }

    .page {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 42px 0;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 28px;
      align-items: end;
      margin-bottom: 26px;
    }

    .badge {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 13px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--muted);
      font-size: 14px;
      backdrop-filter: blur(14px);
    }

    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-3);
      box-shadow: 0 0 18px var(--accent-3);
    }

    h1 {
      margin: 18px 0 12px;
      font-size: clamp(34px, 6vw, 68px);
      line-height: 0.95;
      letter-spacing: -0.07em;
    }

    .hero p {
      max-width: 680px;
      margin: 0;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
    }

    .hero-stat {
      padding: 24px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 28px;
      background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.06));
      box-shadow: 0 24px 70px var(--shadow);
      backdrop-filter: blur(18px);
    }

    .stat-value {
      display: block;
      font-size: 54px;
      font-weight: 800;
      letter-spacing: -0.05em;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      -webkit-background-clip: text;
      color: transparent;
    }

    .stat-label {
      color: var(--muted);
      line-height: 1.5;
    }

    .app {
      display: grid;
      grid-template-columns: 0.92fr 1.08fr;
      gap: 24px;
    }

    .panel {
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 30px;
      background: rgba(15, 23, 42, 0.62);
      box-shadow: 0 24px 70px var(--shadow);
      backdrop-filter: blur(20px);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 24px 24px 0;
    }

    .panel h2 {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }

    .hint {
      color: var(--muted);
      font-size: 14px;
    }

    form {
      padding: 22px 24px 24px;
    }

    .choice-grid {
      display: grid;
      gap: 12px;
    }

    .choice-card {
      position: relative;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 14px;
      align-items: start;
      padding: 16px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.06);
      cursor: pointer;
      transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }

    .choice-card:hover,
    .choice-card.is-selected {
      transform: translateY(-2px);
      border-color: rgba(56, 189, 248, 0.62);
      background: rgba(56, 189, 248, 0.13);
    }

    .choice-card input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    .choice-number {
      display: inline-grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.1);
      color: var(--accent);
      font-weight: 800;
      font-size: 13px;
    }

    .choice-text {
      color: #e2e8f0;
      line-height: 1.45;
    }

    .custom-query {
      display: none;
      margin-top: 16px;
    }

    .custom-query.is-visible {
      display: block;
    }

    textarea {
      width: 100%;
      min-height: 112px;
      resize: vertical;
      padding: 16px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 18px;
      background: rgba(2, 6, 23, 0.45);
      color: var(--text);
      outline: none;
      font: inherit;
      line-height: 1.5;
      transition: border-color 180ms ease, box-shadow 180ms ease;
    }

    textarea:focus {
      border-color: rgba(56, 189, 248, 0.75);
      box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.12);
    }

    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 18px;
    }

    button {
      border: 0;
      border-radius: 16px;
      padding: 14px 18px;
      color: #06121f;
      background: linear-gradient(135deg, var(--accent), var(--accent-3));
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 18px 40px rgba(56, 189, 248, 0.24);
      transition: transform 180ms ease, filter 180ms ease;
    }

    button:hover {
      transform: translateY(-2px);
      filter: brightness(1.06);
    }

    button:disabled {
      cursor: wait;
      filter: grayscale(0.3);
      opacity: 0.8;
    }

    .small-note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .results-panel {
      min-height: 100%;
    }

    .results {
      display: grid;
      gap: 14px;
      padding: 22px 24px 24px;
    }

    .empty-state {
      display: grid;
      place-items: center;
      min-height: 410px;
      padding: 28px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed rgba(255, 255, 255, 0.18);
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.04);
    }

    .empty-state strong {
      display: block;
      margin-bottom: 8px;
      color: var(--text);
      font-size: 22px;
      letter-spacing: -0.03em;
    }

    .query-pill {
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      margin-bottom: 4px;
      padding: 10px 13px;
      border-radius: 999px;
      background: rgba(167, 139, 250, 0.16);
      color: #ddd6fe;
      font-size: 14px;
      line-height: 1.4;
    }

    .result-card {
      padding: 18px;
      border: 1px solid rgba(255, 255, 255, 0.13);
      border-radius: 22px;
      background: linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.05));
      animation: rise 320ms ease both;
    }

    .result-top {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 12px;
    }

    .rank {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--accent);
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .score {
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(52, 211, 153, 0.14);
      color: #bbf7d0;
      font-size: 13px;
      white-space: nowrap;
    }

    .result-text {
      margin: 0;
      color: #f1f5f9;
      font-size: 17px;
      line-height: 1.65;
    }

    .error {
      color: #fecdd3;
      background: rgba(251, 113, 133, 0.14);
      border-color: rgba(251, 113, 133, 0.38);
    }

    @keyframes rise {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 860px) {
      .hero,
      .app {
        grid-template-columns: 1fr;
      }

      .page {
        padding: 26px 0;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <span class="badge"><span class="badge-dot"></span> FAISS + __BACKEND_LABEL__</span>
        <h1>Explorer l’inflation tunisienne .</h1>
        <p> lance la recherche, puis vois les passages les plus proches dans le corpus. Tu peux aussi poser ta propre question.</p>
      </div>
      <aside class="hero-stat" aria-label="Résumé du corpus">
        <span class="stat-value">10</span>
        <span class="stat-label">phrases indexées sur les prix, les secteurs et les contributions à l’inflation.</span>
      </aside>
    </section>

    <section class="app">
      <div class="panel">
        <div class="panel-header">
          <h2>Choisir une requête</h2>
          <span class="hint">6 choix</span>
        </div>
        <form id="search-form">
          <div class="choice-grid">
            __THEME_CARDS__
          </div>

          <div class="custom-query" id="custom-query-wrap">
            <label class="hint" for="custom-query">Ta question personnalisée</label>
            <textarea id="custom-query" placeholder="Exemple : Quels prix ont le plus augmenté ?"></textarea>
          </div>

          <div class="actions">
            <button id="search-button" type="submit">Voir les résultats</button>
            <span class="small-note">Résultats classés par proximité sémantique.</span>
          </div>
        </form>
      </div>

      <div class="panel results-panel">
        <div class="panel-header">
          <h2>Résultats</h2>
          <span class="hint">Top 3</span>
        </div>
        <div class="results" id="results">
          <div class="empty-state">
            <div>
              <strong>Prête quand tu l’es ✨</strong>
              Sélectionne un thème à gauche pour afficher les meilleurs passages.
            </div>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const themes = __THEMES_JSON__;
    const form = document.querySelector("#search-form");
    const cards = Array.from(document.querySelectorAll(".choice-card"));
    const customWrap = document.querySelector("#custom-query-wrap");
    const customQuery = document.querySelector("#custom-query");
    const results = document.querySelector("#results");
    const button = document.querySelector("#search-button");

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function selectedChoice() {
      return document.querySelector('input[name="theme"]:checked');
    }

    function updateSelection() {
      const selected = selectedChoice();
      cards.forEach((card) => {
        const input = card.querySelector("input");
        card.classList.toggle("is-selected", input.checked);
      });
      customWrap.classList.toggle("is-visible", selected?.value === "6");
      if (selected?.value === "6") {
        customQuery.focus();
      }
    }

    function setLoading(isLoading) {
      button.disabled = isLoading;
      button.textContent = isLoading ? "Recherche..." : "Voir les résultats";
    }

    function renderEmpty(message, isError = false) {
      results.innerHTML = `
        <div class="empty-state ${isError ? "error" : ""}">
          <div>
            <strong>${isError ? "Oups, petit caillou dans la machine" : "Aucun résultat"}</strong>
            ${escapeHtml(message)}
          </div>
        </div>
      `;
    }

    function renderResults(payload) {
      const query = escapeHtml(payload.query);
      const cardsHtml = payload.results.map((item) => `
        <article class="result-card">
          <div class="result-top">
            <span class="rank">#${item.rank} résultat</span>
            <span class="score">${item.score}% proche · distance ${item.distance.toFixed(4)}</span>
          </div>
          <p class="result-text">${escapeHtml(item.text)}</p>
        </article>
      `).join("");

      results.innerHTML = `
        <span class="query-pill">Recherche : ${query}</span>
        ${cardsHtml}
      `;
    }

    cards.forEach((card) => {
      card.addEventListener("click", () => {
        card.querySelector("input").checked = true;
        updateSelection();
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const selected = selectedChoice();
      const query = selected?.value === "6" ? customQuery.value.trim() : themes[selected?.value];

      if (!query) {
        renderEmpty("Écris une question personnalisée avant de lancer la recherche.", true);
        return;
      }

      setLoading(true);
      try {
        const response = await fetch("/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "La recherche a échoué.");
        }
        renderResults(payload);
      } catch (error) {
        renderEmpty(error.message, true);
      } finally {
        setLoading(false);
      }
    });

    updateSelection();
  </script>
</body>
</html>
"""


def build_theme_cards():
    cards = []
    for key, theme in THEMES.items():
        checked = " checked" if key == "1" else ""
        cards.append(
            f"""
            <label class="choice-card">
              <input type="radio" name="theme" value="{escape(key)}"{checked}>
              <span class="choice-number">{escape(key).zfill(2)}</span>
              <span class="choice-text">{escape(theme)}</span>
            </label>
            """
        )
    return "\n".join(cards)


def render_page():
    return (
        HTML_TEMPLATE
        .replace("__THEME_CARDS__", build_theme_cards())
        .replace("__THEMES_JSON__", json.dumps(THEMES, ensure_ascii=False))
        .replace("__BACKEND_LABEL__", escape(backend_label))
    )


def semantic_search(query, result_count=RESULT_COUNT):
    clean_query = query.strip()
    if not clean_query:
        return []

    query_vector = model.encode([clean_query], convert_to_numpy=True).astype("float32")
    distances, indexes = index.search(query_vector, k=min(result_count, len(CORPUS)))

    matches = []
    for rank, (idx, distance) in enumerate(zip(indexes[0], distances[0]), start=1):
        distance_value = float(distance)
        matches.append(
            {
                "rank": rank,
                "text": CORPUS[int(idx)],
                "distance": distance_value,
                "score": round(100 / (1 + distance_value), 1),
            }
        )
    return matches


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html(render_page())
            return

        if path == "/health":
            self.send_json({"status": "ok"})
            return

        self.send_json({"error": "Page introuvable."}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/search":
            self.send_json({"error": "Route introuvable."}, status=404)
            return

        payload = self.read_json_body()
        query = payload.get("query", "").strip() if isinstance(payload, dict) else ""
        if not query:
            self.send_json({"error": "La requête ne peut pas être vide."}, status=400)
            return

        self.send_json({"query": query, "results": semantic_search(query)})

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def run_server():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer((host, port), SearchHandler)
    print(f"Page prête : http://{host}:{port}")
    print("Appuyez sur Ctrl+C pour arrêter le serveur.")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
