"""Client LLM unique pour tous les agents. Swappable OpenAI <-> vLLM self-hosted (MI300X)."""
import os, json, urllib.request

# Par défaut OpenAI ($50 crédits hacka). Pour basculer sur le vLLM MI300X :
#   export LLM_BASE_URL=http://<ip-box>:8000/v1 ; export LLM_MODEL=<model-id>
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
API_KEY  = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", "sk-none"))


def chat(system: str, user: str, model: str | None = None, temperature: float = 0.2) -> str:
    """Un appel LLM. Renvoie le texte de la réponse."""
    body = {
        "model": model or MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def chat_json(system: str, user: str, **kw) -> dict:
    """Comme chat() mais parse un JSON dans la réponse (tolérant aux ```json fences)."""
    txt = chat(system, user + "\n\nRéponds UNIQUEMENT en JSON valide.", **kw)
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(txt)
    except Exception:
        # fallback : extrait le premier objet {...}
        i, j = txt.find("{"), txt.rfind("}")
        return json.loads(txt[i:j+1]) if i >= 0 else {"_raw": txt}
