# Proyecto Personal: Registro de Presion Arterial con Telegram + OCR Local + Google Sheets

## Objetivo

Automatizar el registro diario de mediciones de presion arterial a partir de una foto del tensiometro enviada por Telegram, para guardar los datos automaticamente en Google Sheets.

## Alcance

- Uso exclusivamente personal.
- Sin interfaz web para usuarios.
- Solo un servicio backend con webhook publico para Telegram.
- Despliegue en Railway usando la URL que provee el proveedor.

## Flujo funcional

1. Te tomas la presion arterial.
2. Sacas una foto de la pantalla del aparato.
3. Envias la foto al bot de Telegram.
4. Telegram envia el evento al webhook desplegado en Railway.
5. El backend descarga la imagen desde Telegram.
6. Un OCR local extrae los datos de la foto en formato estructurado.
7. El backend valida los valores y registra una nueva fila en Google Sheets con fecha y hora.
8. El bot responde por Telegram confirmando el guardado.

## Datos sugeridos en Google Sheets

Columnas recomendadas:

- timestamp
- sistolica
- diastolica
- pulso
- origen
- confianza_ia
- estado
- observacion
- telegram_file_id

## Arquitectura tecnica (MVP)

- Backend: Python + FastAPI
- Mensajeria: Telegram Bot API (webhook)
- IA OCR/Extraccion: OCR local (Tesseract)
- Persistencia: Google Sheets API
- Hosting: Railway

## Variables de entorno

Definir en Railway:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_WEBHOOK_SECRET
- GOOGLE_SHEETS_ID
- GOOGLE_SERVICE_ACCOUNT_JSON
- TZ=America/Argentina/Buenos_Aires

Opcionales (recomendado):

- TESSERACT_CMD=/usr/bin/tesseract (si el binario no esta en PATH)
- PROCESSING_TIMEOUT_SECONDS=150
- SHEETS_TIMEOUT_SECONDS=30

Notas:

- GOOGLE_SERVICE_ACCOUNT_JSON debe contener el JSON completo de la service account en una sola variable.
- Compartir el Google Sheet con el email de la service account (permiso editor).
- Si notas demoras o timeouts, ajustar los valores opcionales de timeout segun tu red/hosting.

## Seguridad minima recomendada

- Validar el secreto del webhook de Telegram en cada request.
- No exponer claves en codigo ni en repositorio.
- Rechazar mensajes que no contengan foto.
- Registrar logs sin datos sensibles.

## Validaciones utiles

Antes de guardar en Sheets:

- Verificar que existan sistolica, diastolica y pulso.
- Validar que sean enteros positivos.
- Validar rango razonable (ejemplo):
  - sistolica: 70 a 250
  - diastolica: 40 a 150
  - pulso: 30 a 220

Si la extraccion no es confiable:

- Marcar estado=pendiente_revision.
- Responder por Telegram pidiendo confirmacion manual.

## Nivel de complejidad

Complejidad general: media.

Motivo:

- La logica de negocio es simple.
- La complejidad real esta en integrar y estabilizar 3 servicios (Telegram, OCR local y Sheets) + dependencias de sistema.

## Estimacion de tiempos

- MVP funcional: 4 a 6 horas.
- MVP robusto (manejo de errores, validaciones y logs): 1 a 2 dias.
- Version estable para uso continuo: 2 a 3 dias.

## Plan de implementacion sugerido

1. Crear bot de Telegram y obtener token.
2. Crear Google Sheet y service account.
3. Implementar backend FastAPI con endpoint de salud y webhook.
4. Integrar descarga de imagen desde Telegram.
5. Integrar OCR local para extraer valores del tensiometro.
6. Integrar escritura en Google Sheets.
7. Desplegar en Railway y configurar variables.
8. Registrar webhook de Telegram contra URL de Railway.
9. Probar flujo de punta a punta con fotos reales.
10. Ajustar prompt y validaciones para mejorar precision.

## Criterios de exito del MVP

- Enviar una foto al bot.
- Recibir confirmacion de registro por Telegram.
- Ver fila nueva en Google Sheets con valores correctos y timestamp.

## Riesgos y mitigacion

- Lectura OCR incorrecta por calidad de foto:
  - Mitigacion: prompt estricto, validaciones, confirmacion manual cuando la confianza sea baja.
- Falta del binario de Tesseract en el servidor:
  - Mitigacion: instalar Tesseract en el entorno de despliegue y validar en startup.
- Errores de permisos en Sheets:
  - Mitigacion: compartir hoja con service account y probar con insercion manual inicial.
- Webhook no llega por mala configuracion:
  - Mitigacion: endpoint de salud, logs y prueba de setWebhook.

## Entregable final esperado

Un servicio desplegado en Railway (sin frontend) que reciba fotos desde Telegram, extraiga datos de presion arterial con OCR local y registre automaticamente cada medicion en Google Sheets con fecha y hora.

## Estado actual

Ya esta creada la base del proyecto con:

- API FastAPI en `app/main.py`
- Dependencias en `requirements.txt`
- Variables de entorno de ejemplo en `.env.example`
- Comando de arranque para Railway en `Procfile`

## Puesta en marcha local

1. Crear y activar entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Cargar variables de entorno (manual o con tu metodo preferido).

4. Ejecutar API:

```powershell
uvicorn app.main:app --reload --port 8000
```

5. Verificar salud:

```powershell
curl http://localhost:8000/health
```

## Despliegue en Railway

1. Subir este repo a GitHub.
2. Crear proyecto en Railway desde el repo.
3. Configurar variables de entorno con los nombres definidos en este README.
4. Deploy (Railway detecta `Procfile` y levanta uvicorn).
5. Copiar URL publica final, por ejemplo:

```text
https://tu-servicio.up.railway.app
```

## Registrar webhook de Telegram

Con la URL de Railway y tu secreto:

```powershell
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://tu-servicio.up.railway.app/webhook/telegram&secret_token=<TELEGRAM_WEBHOOK_SECRET>"
```

Opcional: validar webhook actual:

```powershell
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo"
```

## Prueba de punta a punta

1. Enviar foto del tensiometro al bot.
2. Esperar mensaje de confirmacion en Telegram.
3. Verificar nueva fila en Google Sheets.

## Nota tecnica importante

El proyecto usa OCR local con Tesseract a traves de `pytesseract`. En el entorno de despliegue debe estar instalado el binario de Tesseract para que el OCR funcione.
