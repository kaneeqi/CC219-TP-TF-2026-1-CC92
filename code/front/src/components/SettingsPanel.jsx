import { MODELO_LABELS, MODELO_CLASICO_RECOMENDADO } from "../intenciones";

function etiquetaOpcion(clave, tarea) {
  const base = MODELO_LABELS[clave] || clave;
  if (clave !== "beto" && MODELO_CLASICO_RECOMENDADO[tarea] === clave) {
    return `${base} (recomendado)`;
  }
  return base;
}

export default function SettingsPanel({
  abierto,
  onCerrar,
  modelosDisponibles,
  modeloIntencion,
  modeloSentimiento,
  onCambiarModeloIntencion,
  onCambiarModeloSentimiento,
}) {
  return (
    <>
      <div className={`overlay ${abierto ? "overlay--visible" : ""}`} onClick={onCerrar} />
      <aside className={`panel ${abierto ? "panel--abierto" : ""}`}>
        <div className="panel__header">
          <h2>Modelos</h2>
          <button className="panel__cerrar" onClick={onCerrar} aria-label="Cerrar">
            ✕
          </button>
        </div>

        <div className="panel__seccion">
          <label htmlFor="modelo-intencion">Modelo de intención</label>
          <select
            id="modelo-intencion"
            value={modeloIntencion}
            onChange={(e) => onCambiarModeloIntencion(e.target.value)}
          >
            {Object.keys(modelosDisponibles.intencion || {}).map((clave) => (
              <option key={clave} value={clave}>
                {etiquetaOpcion(clave, "intencion")}
              </option>
            ))}
          </select>
        </div>

        <div className="panel__seccion">
          <label htmlFor="modelo-sentimiento">Modelo de sentimiento</label>
          <select
            id="modelo-sentimiento"
            value={modeloSentimiento}
            onChange={(e) => onCambiarModeloSentimiento(e.target.value)}
          >
            {Object.keys(modelosDisponibles.sentimiento || {}).map((clave) => (
              <option key={clave} value={clave}>
                {etiquetaOpcion(clave, "sentimiento")}
              </option>
            ))}
          </select>
        </div>

        <p className="panel__nota">
          Intención y sentimiento son dos clasificadores independientes,
          entrenados sobre datasets distintos — puedes combinar cualquier par
          de modelos para comparar resultados. "Recomendado" marca el mejor
          modelo clásico (sin GPU) de cada tarea, medido contra datos reales.
        </p>
      </aside>
    </>
  );
}
