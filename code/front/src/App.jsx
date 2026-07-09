import { useEffect, useRef, useState } from "react";
import MessageBubble from "./components/MessageBubble";
import SettingsPanel from "./components/SettingsPanel";
import { analizarMensaje, obtenerModelos } from "./api";
import logo from "./assets/logo.png";
import "./App.css";

function horaActual() {
  return new Date().toLocaleTimeString("es-PE", { hour: "2-digit", minute: "2-digit" });
}

export default function App() {
  const [mensajes, setMensajes] = useState([]);
  const [texto, setTexto] = useState("");
  const [panelAbierto, setPanelAbierto] = useState(false);
  const [modelosDisponibles, setModelosDisponibles] = useState({ intencion: {}, sentimiento: {} });
  const [modeloIntencion, setModeloIntencion] = useState("beto");
  const [modeloSentimiento, setModeloSentimiento] = useState("beto");
  const [errorConexion, setErrorConexion] = useState(false);
  const finRef = useRef(null);

  useEffect(() => {
    obtenerModelos()
      .then((data) => setModelosDisponibles(data))
      .catch(() => setErrorConexion(true));
  }, []);

  useEffect(() => {
    finRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [mensajes]);

  async function enviarMensaje(e) {
    e.preventDefault();
    const texto_ = texto.trim();
    if (!texto_) return;
    setTexto("");

    const id = crypto.randomUUID();
    setMensajes((prev) => [
      ...prev,
      { id, texto: texto_, hora: horaActual(), estado: "cargando" },
    ]);

    try {
      const analisis = await analizarMensaje(texto_, modeloIntencion, modeloSentimiento);
      setMensajes((prev) =>
        prev.map((m) => (m.id === id ? { ...m, estado: "listo", analisis } : m))
      );
    } catch (err) {
      setMensajes((prev) =>
        prev.map((m) => (m.id === id ? { ...m, estado: "error", error: err.message } : m))
      );
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar__header">
          <div className="sidebar__avatar">
            <img src={logo} alt="Fonazo" />
          </div>
          <span className="sidebar__titulo">Fonazo</span>
        </div>
        <div className="sidebar__lista">
          <div className="sidebar__chat sidebar__chat--activo">
            <div className="sidebar__chat-avatar">
              <img src={logo} alt="Fonazo" />
            </div>
            <div className="sidebar__chat-info">
              <strong>Fonazo — Asistente Analítico</strong>
              <span>Intención · Sentimiento · Urgencia</span>
            </div>
          </div>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="header">
          <div className="header__avatar">
            <img src={logo} alt="Fonazo" />
          </div>
          <div className="header__info">
            <h1>Fonazo — Asistente Analítico</h1>
            <span className="header__subtitulo">Intención · Sentimiento · Urgencia</span>
          </div>
          <button className="header__ajustes" onClick={() => setPanelAbierto(true)} aria-label="Configurar modelos">
            ⚙️
          </button>
        </header>

        {errorConexion && (
          <div className="banner-error">
            No se pudo conectar con la API en {import.meta.env.VITE_API_URL || "http://localhost:8000"}.
            Verifica que el backend esté corriendo.
          </div>
        )}

        <main className="chat">
          {mensajes.length === 0 && (
            <div className="chat__vacio">
              Escribe un mensaje como si le escribieras a la tienda por WhatsApp
              — el sistema detecta la intención, el sentimiento y si requiere
              atención urgente.
            </div>
          )}
          {mensajes.map((m) => (
            <MessageBubble key={m.id} mensaje={m} />
          ))}
          <div ref={finRef} />
        </main>

        <form className="input-bar" onSubmit={enviarMensaje}>
          <input
            type="text"
            placeholder="Escribe un mensaje…"
            value={texto}
            onChange={(e) => setTexto(e.target.value)}
          />
          <button type="submit" aria-label="Enviar" disabled={!texto.trim()}>
            ➤
          </button>
        </form>
      </section>

      <SettingsPanel
        abierto={panelAbierto}
        onCerrar={() => setPanelAbierto(false)}
        modelosDisponibles={modelosDisponibles}
        modeloIntencion={modeloIntencion}
        modeloSentimiento={modeloSentimiento}
        onCambiarModeloIntencion={setModeloIntencion}
        onCambiarModeloSentimiento={setModeloSentimiento}
      />
    </div>
  );
}
