import requests
import base64
import json
from modules.voice_module import falar

API_KEY = "AIzaSyBG88TfZ_IA3bim0oROt6ZDhf9bkLJNxjg"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

def interpretar_imagem_com_openai(caminho_imagem):
    try:
        # Lê e codifica a imagem
        with open(caminho_imagem, "rb") as img_file:
            imagem_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Analise tecnicamente o que aparece nesta imagem. Seja claro, objetivo e ajude como se estivesse explicando para um técnico de manutenção."
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": imagem_base64
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(GEMINI_ENDPOINT, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            resultado = response.json()
            mensagem = resultado["candidates"][0]["content"]["parts"][0]["text"]
            falar(mensagem)
            return mensagem
        else:
            erro = response.text
            print(f"[ERRO] Resposta da API Gemini: {response.status_code} - {erro}")
            falar("Tive um problema ao tentar interpretar a imagem.")
            return f"Erro na resposta da API Gemini: {erro}"

    except Exception as e:
        print(f"[EXCEÇÃO] {e}")
        falar("Tive um erro inesperado ao interpretar a imagem.")
        return f"Erro ao interpretar imagem: {str(e)}"
