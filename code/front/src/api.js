const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function manejarRespuesta(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status} al consultar la API`);
  }
  return res.json();
}

export async function analizarMensaje(texto, modeloIntencion, modeloSentimiento) {
  const res = await fetch(`${API_BASE}/analizar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      texto,
      modelo_intencion: modeloIntencion,
      modelo_sentimiento: modeloSentimiento,
    }),
  });
  return manejarRespuesta(res);
}

export async function obtenerModelos() {
  const res = await fetch(`${API_BASE}/modelos`);
  return manejarRespuesta(res);
}
