import os
import time
import re
import logging
from contextlib import asynccontextmanager
from typing import List
from io import BytesIO

from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from transformers import pipeline
from deep_translator import GoogleTranslator
from pypdf import PdfReader

# --- CONFIGURACIÓN DE AUDITORÍA AVANZADA (LOGGING) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClinicalNLPEngine")

# --- RECURSOS GLOBALES DE IA (MAPPED IN MEMORY) ---
recursos_ia = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación y carga de pesos neuronales."""
    try:
        logger.info("Iniciando secuencia de carga de infraestructura clínica...")
        
        # Traductores base de Deep Translator
        recursos_ia["trad_es_en"] = GoogleTranslator(source='es', target='en')
        recursos_ia["trad_en_es"] = GoogleTranslator(source='en', target='es')
        
        # Pipeline del Transformer Biomédico optimizado para RAM en CPU
        logger.info("Cargando modelo BioBERT (d4data/biomedical-ner-all) en RAM...")
        recursos_ia["biobert"] = pipeline("ner", model="d4data/biomedical-ner-all")
        
        logger.info("Infraestructura MLOps desplegada correctamente en Railway Hobby.")
        yield
    except Exception as e:
        logger.error(f"Fallo crítico insalvable en el inicio del backend: {e}")
        yield
    finally:
        logger.info("Liberando recursos de memoria RAM...")
        recursos_ia.clear()

# --- INSTANCIA DE LA API CON METADATOS ACADÉMICOS ---
app = FastAPI(
    title="Clinical NLP & Heuristic Diagnostic Engine",
    description="API de grado de producción para el procesamiento de lenguaje natural biomédico y extracción de entidades.",
    version="1.2.0",
    contact={
        "name": "Departamento de Ingeniería de Sistemas",
        "email": "soporte@clinicalnlp.edu"
    },
    lifespan=lifespan
)

# Configuración Dinámica y Segura de CORS
cors_env = os.getenv("CORS_ORIGINS", "*")
origins = [cors_env] if cors_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MAPEO DE ONTOLOGÍA MÉDICA (BIOBERT A ESPAÑOL) ---
DICCIONARIO_ETIQUETAS = {
    "Age": "Edad", "Sex": "Sexo", "Sign_symptom": "Signo o Síntoma",
    "Diagnostic_procedure": "Procedimiento Diagnóstico", "Lab_value": "Valor de Laboratorio",
    "Dosage": "Dosis", "Medication": "Medicamento", "Biological_structure": "Estructura Biológica",
    "Nonbiological_location": "Ubicación", "Clinical_event": "Evento Clínico",
    "Detailed_description": "Descripción Detallada", "Family_history": "Antecedente Familiar",
    "History": "Historial Médico"
}

# --- ENTIENTIDADES TIPO (SCHEMAS PYDANTIC) ---
class NotaClinicaIn(BaseModel):
    texto_clinico: str = Field(
        ..., 
        min_length=10, 
        max_length=2000,
        description="Texto libre de la anamnesis o evolución médica del paciente."
    )

class EntidadClinica(BaseModel):
    palabra: str = Field(..., description="Término o concepto médico extraído.")
    etiqueta: str = Field(..., description="Categoría asignada por la ontología médica.")
    certeza: float = Field(..., description="Nivel de confianza estadística del modelo.")

class TelemetriaMeta(BaseModel):
    tiempo_traduccion_ms: float = Field(..., description="Tiempo consumido en normalización lingüística de ida y vuelta.")
    tiempo_inferencia_ms: float = Field(..., description="Tiempo de cómputo neto en la red neuronal BioBERT.")
    tiempo_heuristica_ms: float = Field(..., description="Tiempo de evaluación del motor de reglas cuantitativas.")
    tiempo_total_ms: float = Field(..., description="Latencia de procesamiento extremo a extremo.")

class AnalisisClinicoOut(BaseModel):
    estado: str = Field("exito", description="Estado operativo de la solicitud.")
    alertas: List[str] = Field(..., description="Alertas predictivas disparadas por el motor de reglas.")
    entidades: List[EntidadClinica] = Field(..., description="Lista ordenada de entidades biomédicas.")
    meta: TelemetriaMeta = Field(..., description="Métricas de rendimiento e ingeniería de la petición.")


# --- MOTOR DE REGLAS HEURÍSTICAS AVANZADO ---
class ClinicalRuleEngine:
    """Clase encargada del cribado y normalización de constantes vitales en texto libre."""
    
    @staticmethod
    def evaluar_signos_vitales(texto: str) -> List[str]:
        alertas = []
        fmt_texto = texto.lower()

        # Patrón A: Saturación de Oxígeno (SpO2) con tolerancia sintáctica
        sat_match = re.search(r'(saturaci[óo]n|sat|o2|ox[ií]geno)\s*(de|del)?\s*(\d{2})\s*%?', fmt_texto)
        if sat_match:
            valor = int(sat_match.group(3))
            if valor < 90:
                alertas.append(f"⚠️ Alerta de Hipoxia Severa ({valor}%): Riesgo clínico alto. Evaluar soporte ventilatorio.")

        # Patrón B: Frecuencia Cardíaca (FC)
        fc_match = re.search(r'(\d{2,3})\s*(lpm|frecuencia card[ií]aca|pulso|latidos|f\.c)', fmt_texto)
        if fc_match:
            valor = int(fc_match.group(1))
            if valor > 100:
                alertas.append(f"⚠️ Alerta de Taquicardia ({valor} lpm): Ritmo cardíaco acelerado detectado.")
            elif valor < 60:
                alertas.append(f"⚠️ Alerta de Bradicardia ({valor} lpm): Ritmo cardíaco inferior a los rangos normales.")

        # Patrón C: Temperatura Corporal
        temp_match = re.search(r'(\d{2}(\.\d)?)\s*(°c|grados|temperatura|temp)', fmt_texto)
        if temp_match:
            valor = float(temp_match.group(1))
            if valor >= 38.0:
                alertas.append(f"⚠️ Alerta de Síndrome Febril ({valor} °C): Proceso pirético activo.")

        # Patrón D: Presión Arterial (PA)
        pa_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})\s*(mm\s*hg|mmhg|presi[oó]n)?', fmt_texto)
        if pa_match:
            sistolica = int(pa_match.group(1))
            diastolica = int(pa_match.group(2))
            if sistolica > 140 or diastolica > 90:
                alertas.append(f"⚠️ Riesgo Cardiovascular ({sistolica}/{diastolica} mmHg): Valores sugerentes de Crisis Hipertensiva.")

        return alertas


# --- ALGORITMO DE RECONSTRUCCIÓN DE TOKENS BIO (MÁQUINA DE ESTADOS) ---
def reconstruir_tokens_biomedicos(resultados_crudos) -> List[dict]:
    entidades = []
    actual = None

    for res in resultados_crudos:
        palabra = res['word']
        es_subtoken = palabra.startswith("##")
        palabra_limpia = palabra.replace("##", "")
        
        etiqueta_cruda = res.get('entity_group', res.get('entity', 'Unknown'))
        etiq_limpia = etiqueta_cruda.replace("B-", "").replace("I-", "")
        certeza = float(res['score'])

        if actual is None:
            actual = {'palabra': palabra_limpia, 'etiqueta': etiq_limpia, 'certeza': certeza}
        elif etiq_limpia == actual['etiqueta'] or es_subtoken:
            separador = "" if es_subtoken else " "
            actual['palabra'] += f"{separador}{palabra_limpia}"
            actual['certeza'] = min(actual['certeza'], certeza)
        else:
            entidades.append(actual)
            actual = {'palabra': palabra_limpia, 'etiqueta': etiq_limpia, 'certeza': certeza}

    if actual: 
        entidades.append(actual)
        
    return entidades


# --- ENDPOINT 1: PROCESAMIENTO DE TEXTO CLÍNICO DIRECTO ---
@app.post(
    "/api/v1/analizar-texto", 
    response_model=AnalisisClinicoOut,
    status_code=status.HTTP_200_OK,
    summary="Procesa notas de evolución clínica en texto libre."
)
async def analizar_texto_clinico(solicitud: NotaClinicaIn):
    if not recursos_ia:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Los motores de inferencia no se inicializaron correctamente."
        )
    
    t_inicio = time.perf_counter()
    
    try:
        # 1. Cómputo del Motor Heurístico (Nativo en CPU)
        t_h_inicio = time.perf_counter()
        alertas_detectadas = ClinicalRuleEngine.evaluar_signos_vitales(solicitud.texto_clinico)
        t_h_fin = time.perf_counter()
        
        # 2. Normalización Lingüística de Entrada (Es -> En) en Threadpool
        t_t_inicio = time.perf_counter()
        texto_en = await run_in_threadpool(recursos_ia["trad_es_en"].translate, solicitud.texto_clinico)
        t_t_fin = time.perf_counter()
        
        # 3. Inferencia de la Red Neuronal BioBERT
        t_i_inicio = time.perf_counter()
        resultados_crudos = recursos_ia["biobert"](texto_en)
        entidades_ensambladas = reconstruir_tokens_biomedicos(resultados_crudos)
        t_i_fin = time.perf_counter()
        
        # 4. TRADUCCIÓN MASIVA EN LOTE (BATCH OPTIMIZATION)
        t_t_back_inicio = time.perf_counter()
        filtradas = [e for e in entidades_ensambladas if e['certeza'] >= 0.45 and e['etiqueta'] != "O"]
        palabras_en = [e['palabra'] for e in filtradas]
        
        palabras_es = []
        if palabras_en:
            # Una sola llamada HTTP masiva para mitigar la latencia de red
            palabras_es = await run_in_threadpool(recursos_ia["trad_en_es"].translate_batch, palabras_en)
        
        entidades_finales = []
        for i, ent in enumerate(filtradas):
            palabra_traducida = palabras_es[i] if i < len(palabras_es) else ent['palabra']
            etiqueta_es = DICCIONARIO_ETIQUETAS.get(ent['etiqueta'], ent['etiqueta'])
            
            entidades_finales.append(
                EntidadClinica(
                    palabra=palabra_traducida,
                    etiqueta=etiqueta_es,
                    certeza=round(ent['certeza'], 4)
                )
            )
        t_t_back_fin = time.perf_counter()
        
        t_final = time.perf_counter()
        
        # Consolidación de métricas de telemetría médica
        tiempo_traduccion_total = ((t_t_fin - t_t_inicio) + (t_t_back_fin - t_t_back_inicio)) * 1000
        
        meta_rendimiento = TelemetriaMeta(
            tiempo_traduccion_ms=round(tiempo_traduccion_total, 2),
            tiempo_inferencia_ms=round((t_i_fin - t_i_inicio) * 1000, 2),
            tiempo_heuristica_ms=round((t_h_fin - t_h_inicio) * 1000, 2),
            tiempo_total_ms=round((t_final - t_inicio) * 1000, 2)
        )
        
        return AnalisisClinicoOut(
            estado="exito",
            alertas=alertas_detectadas,
            entidades=entidades_finales,
            meta=meta_rendimiento
        )
        
    except Exception as e:
        logger.error(f"Fallo en la tubería del pipeline de texto: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error crítico interno procesando la solicitud biomédica."
        )


# --- ENDPOINT 2: EXTRACCIÓN Y PROCESAMIENTO DE PDFS EN MEMORIA ---
@app.post(
    "/api/v1/analizar-pdf", 
    response_model=AnalisisClinicoOut,
    status_code=status.HTTP_200_OK,
    summary="Procesa y extrae entidades desde un archivo PDF digital nativo."
)
async def analizar_pdf_clinico(archivo: UploadFile = File(...)):
    # Validación estricta del tipo MIME
    if archivo.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. El sistema solo acepta documentos con extensión estructural .pdf"
        )
    
    try:
        # Lectura binaria directa sobre RAM (Evita escrituras en el disco efímero de Railway)
        contenido_binario = await archivo.read()
        lector_pdf = PdfReader(BytesIO(contenido_binario))
        
        texto_extraido = ""
        for pagina in lector_pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto_extraido += f"\n{texto_pagina}"
        
        # Limpieza básica de saltos de página y espaciados redundantes
        texto_limpio = re.sub(r'\s+', ' ', texto_extraido).strip()
        
        if len(texto_limpio) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="El PDF no contiene suficiente texto legible. Verifique que no sea un archivo escaneado (imagen)."
            )
        
        # Inyección directa a la lógica optimizada del pipeline principal
        objeto_solicitud = NotaClinicaIn(texto_clinico=texto_limpio)
        return await analizar_texto_clinico(objeto_solicitud)
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Fallo crítico parseando el documento PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al parsear la estructura de flujo del PDF."
        )


# --- ORQUESTADOR DE ENTRADA DINÁMICO ---
if __name__ == "__main__":
    puerto = int(os.getenv("PORT", 8000))
    es_produccion = os.getenv("ENV", "development") == "production"
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=puerto, 
        reload=not es_produccion
    )