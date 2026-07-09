import { INTENCION_LABELS, INTENCION_COLOR } from "../intenciones";

function Chip({ children, style, className = "" }) {
  return (
    <span className={`chip ${className}`} style={style}>
      {children}
    </span>
  );
}

export default function MessageBubble({ mensaje }) {
  const { texto, estado, analisis, error, hora } = mensaje;
  const urgente = analisis?.urgente;

  return (
    <div className={`bubble-row ${urgente ? "bubble-row--urgente" : ""}`}>
      <div className={`bubble ${urgente ? "bubble--urgente" : ""}`}>
        <p className="bubble__texto">{texto}</p>

        {estado === "cargando" && <div className="bubble__cargando">Analizando…</div>}
        {estado === "error" && <div className="bubble__error">⚠ {error}</div>}

        {estado === "listo" && analisis && (
          <div className="chips">
            <Chip style={{ background: INTENCION_COLOR[analisis.intencion.etiqueta] || "#607D8B" }}>
              {INTENCION_LABELS[analisis.intencion.etiqueta] || analisis.intencion.etiqueta}
              {" · "}
              {(analisis.intencion.confianza * 100).toFixed(0)}%
            </Chip>

            {analisis.sentimiento ? (
              <Chip className={`chip--sentimiento-${analisis.sentimiento.etiqueta}`}>
                {analisis.sentimiento.etiqueta === "positivo" ? "🟢" : "🔴"} {analisis.sentimiento.etiqueta}
                {" · "}
                {(analisis.sentimiento.confianza * 100).toFixed(0)}%
              </Chip>
            ) : (
              <Chip className="chip--neutro">⚪ sentimiento no aplica</Chip>
            )}

            {urgente && <Chip className="chip--urgente">🚨 Urgente</Chip>}
          </div>
        )}

        <span className="bubble__hora">{hora}</span>
      </div>
    </div>
  );
}
