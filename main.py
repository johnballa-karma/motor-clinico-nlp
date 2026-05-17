import os
import time
import re
import logging
from contextlib import asynccontextmanager
from typing import List, Callable
from io import BytesIO

from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from transformers import pipeline
from deep_translator import GoogleTranslator
from pypdf import PdfReader
import spacy
from spacy.matcher import Matcher

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
    try:
        logger.info("Iniciando secuencia de carga de infraestructura clínica...")
        
        recursos_ia["trad_es_en"] = GoogleTranslator(source='es', target='en')
        recursos_ia["trad_en_es"] = GoogleTranslator(source='en', target='es')
        
        logger.info("Cargando pipeline lingüístico spaCy (es_core_news_sm)...")
        recursos_ia["nlp_es"] = spacy.load("es_core_news_sm")
        
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

app = FastAPI(
    title="Clinical NLP & Heuristic Diagnostic Engine",
    description="API de grado de producción para el procesamiento de lenguaje natural biomédico.",
    version="1.4.0",
    lifespan=lifespan
)

cors_env = os.getenv("CORS_ORIGINS", "*")
origins = [cors_env] if cors_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DICCIONARIO_ETIQUETAS = {
    "Age": "Edad", "Sex": "Sexo", "Sign_symptom": "Signo o Síntoma",
    "Diagnostic_procedure": "Procedimiento Diagnóstico", "Lab_value": "Valor de Laboratorio",
    "Dosage": "Dosis", "Medication": "Medicamento", "Biological_structure": "Estructura Biológica",
    "Nonbiological_location": "Ubicación", "Clinical_event": "Evento Clínico",
    "Detailed_description": "Descripción Detallada", "Family_history": "Antecedente Familiar",
    "History": "Historial Médico"
}

class NotaClinicaIn(BaseModel):
    texto_clinico: str = Field(..., min_length=10, max_length=100000)

class EntidadClinica(BaseModel):
    palabra: str
    etiqueta: str
    certeza: float

class TelemetriaMeta(BaseModel):
    tiempo_traduccion_ms: float
    tiempo_inferencia_ms: float
    tiempo_heuristica_ms: float
    tiempo_total_ms: float

class AnalisisClinicoOut(BaseModel):
    estado: str = "exito"
    alertas: List[str] = []
    entidades: List[EntidadClinica] = []
    meta: TelemetriaMeta


# --- ARQUITECTURA ESCALABLE DE BIOMARCADORES ---
# Si necesitas agregar nuevas enfermedades derivadas de números (ej. creatinina -> Falla Renal), solo agregas un bloque aquí.
REGLAS_CLINICAS_CUANTITATIVAS = [
    {
        "id": "oxigeno",
        "lemmas": ["saturación", "saturar", "sat", "o2", "oxígeno"],
        "min_valid": 50.0, "max_valid": 100.0,
        "evaluaciones": [
            (lambda v: v < 90.0, "⚠️ Hipoxia Severa ({val}%): Riesgo clínico alto. Evaluar soporte ventilatorio.")
        ]
    },
    {
        "id": "frecuencia_cardiaca",
        "lemmas": ["frecuencia", "latido", "pulso", "lpm", "fc", "cardíaco", "cardiaca"],
        "min_valid": 30.0, "max_valid": 250.0,
        "evaluaciones": [
            (lambda v: v > 100.0, "⚠️ Taquicardia ({val} lpm): Ritmo cardíaco acelerado detectado."),
            (lambda v: v < 60.0, "⚠️ Bradicardia ({val} lpm): Ritmo cardíaco inferior a los rangos normales.")
        ]
    },
    {
        "id": "temperatura",
        "lemmas": ["temperatura", "fiebre", "temp", "°"],
        "min_valid": 35.0, "max_valid": 43.0,
        "evaluaciones": [
            (lambda v: v >= 38.0, "⚠️ Síndrome Febril ({val} °C): Proceso pirético activo.")
        ]
    },
    {
        "id": "glucosa",
        "lemmas": ["glucosa", "glicemia", "azúcar", "mg/dl"],
        "min_valid": 20.0, "max_valid": 1000.0,
        "evaluaciones": [
            (lambda v: v > 125.0, "⚠️ Hiperglucemia ({val}): Niveles sugerentes de Diabetes Mellitus o descompensación."),
            (lambda v: v < 70.0, "⚠️ Hipoglucemia ({val}): Riesgo neurológico agudo.")
        ]
    },
    {
        "id": "hemoglobina",
        "lemmas": ["hemoglobina", "hb"],
        "min_valid": 3.0, "max_valid": 25.0,
        "evaluaciones": [
            (lambda v: v < 12.0, "⚠️ Sospecha de Anemia ({val} g/dL): Disminución de la masa eritrocitaria detectada.")
        ]
    },
    {
        "id": "frecuencia_respiratoria",
        "lemmas": ["respiración", "respiratoria", "rpm", "fr"],
        "min_valid": 5.0, "max_valid": 60.0,
        "evaluaciones": [
            (lambda v: v > 20.0, "⚠️ Taquipnea ({val} rpm): Frecuencia respiratoria elevada."),
            (lambda v: v < 12.0, "⚠️ Bradipnea ({val} rpm): Depresión respiratoria detectada.")
        ]
    }
]

class ClinicalRuleEngine:
    @staticmethod
    def evaluar_signos_vitales(texto: str) -> List[str]:
        alertas = []
        nlp = recursos_ia.get("nlp_es")
        if not nlp:
            return alertas

        doc = nlp(texto.lower())
        matcher = Matcher(nlp.vocab)

        # 1. Patrón Complejo: Presión Arterial (Se mantiene aislado por su sintaxis doble)
        patron_pa = [
            {"IS_DIGIT": True},
            {"LOWER": {"IN": ["/", "sobre", "con", "de", "-"]}, "OP": "?"},
            {"IS_DIGIT": True}
        ]
        matcher.add("PRESION_ARTERIAL", [patron_pa])
        coincidencias = matcher(doc)

        for match_id, start, end in coincidencias:
            sub_doc = doc[start:end]
            numeros = [int(token.text) for token in sub_doc if token.is_digit]
            if len(numeros) == 2:
                sistolica, diastolica = numeros[0], numeros[1]
                if 50 <= sistolica <= 250 and 30 <= diastolica <= 150:
                    if sistolica > 140 or diastolica > 90:
                        alertas.append(f"⚠️ Riesgo Cardiovascular ({sistolica}/{diastolica} mmHg): Valores sugerentes de Crisis Hipertensiva.")

        # 2. Análisis Semántico Escalable (Lee el diccionario dinámicamente)
        for token in doc:
            for regla in REGLAS_CLINICAS_CUANTITATIVAS:
                if token.lemma_ in regla["lemmas"]:
                    # Abrimos una ventana de búsqueda de 2 palabras antes y 4 después del término médico
                    ventana = doc[max(0, token.i - 2) : min(len(doc), token.i + 4)]
                    for t in ventana:
                        if t.is_digit or (t.like_num and ("." in t.text or "," in t.text)):
                            try:
                                val = float(t.text.replace(",", "."))
                                # Si el número tiene coherencia fisiológica para esa prueba médica
                                if regla["min_valid"] <= val <= regla["max_valid"]:
                                    for condicion, mensaje in regla["evaluaciones"]:
                                        if condicion(val):
                                            alertas.append(mensaje.format(val=val))
                                            break # Rompe el ciclo para no repetir la alerta del mismo número
                            except ValueError:
                                continue

        return list(dict.fromkeys(alertas))


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


@app.post("/api/v1/analizar-texto", response_model=AnalisisClinicoOut)
async def analizar_texto_clinico(solicitud: NotaClinicaIn):
    if not recursos_ia:
        raise HTTPException(status_code=503, detail="Motores no inicializados.")
    
    t_inicio = time.perf_counter()
    try:
        t_h_inicio = time.perf_counter()
        alertas_detectadas = ClinicalRuleEngine.evaluar_signos_vitales(solicitud.texto_clinico)
        t_h_fin = time.perf_counter()
        
        t_t_inicio = time.perf_counter()
        texto_en = await run_in_threadpool(recursos_ia["trad_es_en"].translate, solicitud.texto_clinico)
        t_t_fin = time.perf_counter()
        
        t_i_inicio = time.perf_counter()
        resultados_crudos = recursos_ia["biobert"](texto_en)
        entidades_ensambladas = reconstruir_tokens_biomedicos(resultados_crudos)
        t_i_fin = time.perf_counter()
        
        t_t_back_inicio = time.perf_counter()
        filtradas = [e for e in entidades_ensambladas if e['certeza'] >= 0.45 and e['etiqueta'] != "O"]
        palabras_en = [e['palabra'] for e in filtradas]
        
        palabras_es = []
        if palabras_en:
            palabras_es = await run_in_threadpool(recursos_ia["trad_en_es"].translate_batch, palabras_en)
        
        entidades_finales = []
        for i, ent in enumerate(filtradas):
            palabra_traducida = palabras_es[i] if i < len(palabras_es) else ent['palabra']
            etiqueta_es = DICCIONARIO_ETIQUETAS.get(ent['etiqueta'], ent['etiqueta'])
            entidades_finales.append(EntidadClinica(palabra=palabra_traducida, etiqueta=etiqueta_es, certeza=round(ent['certeza'], 4)))
        t_t_back_fin = time.perf_counter()
        
        t_final = time.perf_counter()
        
        meta = TelemetriaMeta(
            tiempo_traduccion_ms=round(((t_t_fin - t_t_inicio) + (t_t_back_fin - t_t_back_inicio)) * 1000, 2),
            tiempo_inferencia_ms=round((t_i_fin - t_i_inicio) * 1000, 2),
            tiempo_heuristica_ms=round((t_h_fin - t_h_inicio) * 1000, 2),
            tiempo_total_ms=round((t_final - t_inicio) * 1000, 2)
        )
        
        return AnalisisClinicoOut(estado="exito", alertas=alertas_detectadas, entidades=entidades_finales, meta=meta)
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        raise HTTPException(status_code=500, detail="Error interno procesando la solicitud.")


@app.post("/api/v1/analizar-pdf", response_model=AnalisisClinicoOut)
async def analizar_pdf_clinico(archivo: UploadFile = File(...)):
    extension = os.path.splitext(archivo.filename)[1].lower()
    if archivo.content_type != "application/pdf" and extension != ".pdf":
        raise HTTPException(status_code=400, detail="Solo se admiten archivos .pdf reales.")
    
    try:
        lector_pdf = PdfReader(BytesIO(await archivo.read()))
        texto_extraido = "\n".join([pagina.extract_text() for pagina in lector_pdf.pages if pagina.extract_text()])
        texto_limpio = re.sub(r'\s+', ' ', texto_extraido).strip()
        
        if len(texto_limpio) < 10:
            raise HTTPException(status_code=422, detail="PDF sin texto legible.")
        
        return await analizar_texto_clinico(NotaClinicaIn(texto_clinico=texto_limpio))
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el backend: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    es_produccion = os.getenv("ENV", "development") == "production"
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=not es_produccion)