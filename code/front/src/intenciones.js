// Etiquetas legibles y color por categoria de intencion, para los chips de
// analisis en el chat. Vive aparte para no mezclar datos de presentacion
// con la logica de los componentes.
export const INTENCION_LABELS = {
  consulta_producto: "Consulta de producto",
  consulta_precio: "Consulta de precio",
  consulta_servicio_tecnico: "Consulta técnica",
  queja_producto_servicio: "Queja",
  solicitud_garantia_cambio: "Garantía / cambio",
  seguimiento_reparacion: "Seguimiento de reparación",
  escalamiento_urgente: "Escalamiento urgente",
  saludo_cierre: "Saludo / cierre",
};

export const INTENCION_COLOR = {
  consulta_producto: "#2196F3",
  consulta_precio: "#00897B",
  consulta_servicio_tecnico: "#7E57C2",
  queja_producto_servicio: "#FB8C00",
  solicitud_garantia_cambio: "#C79100",
  seguimiento_reparacion: "#00ACC1",
  escalamiento_urgente: "#E53935",
  saludo_cierre: "#43A047",
};

export const MODELO_LABELS = {
  beto: "BETO (fine-tuned)",
  logreg: "Regresión Logística",
  naive_bayes: "Naive Bayes",
  svm: "SVM Lineal",
  xgboost: "XGBoost",
};

// Mejor modelo clasico por tarea, medido contra el holdout real / validacion
// fuera de plantilla (ver PLAN_REDISENO.md). BETO ya es el default de ambas
// tareas -- esto es para saber cual clasico probar si se quiere algo
// mas liviano/sin GPU.
export const MODELO_CLASICO_RECOMENDADO = {
  sentimiento: "logreg",
  intencion: "svm",
};
