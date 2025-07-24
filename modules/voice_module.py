# modules/voice_module.py
import speech_recognition as sr
import pyttsx3

# Inicializa o motor de fala
engine = pyttsx3.init()
engine.setProperty('rate', 220)  # Velocidade da fala

# Configura voz masculina "Microsoft Daniel" se estiver disponível
voices = engine.getProperty('voices')
for voice in voices:
    if "daniel" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

reconhecedor = sr.Recognizer()

def falar(texto):
    print(f"[EVA] {texto}")
    try:
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        print(f"[ERRO NA FALA] {e}")

def reconhecer_usuario():
    with sr.Microphone() as source:
        print("[🎤] Aguardando sua voz...")
        reconhecedor.adjust_for_ambient_noise(source, duration=1.5)  # melhor calibração do ruído
        try:
            audio = reconhecedor.listen(source, timeout=5, phrase_time_limit=7)  # mais controle
        except sr.WaitTimeoutError:
            falar("Não ouvi nada. Pode tentar de novo?")
            return reconhecer_usuario()

    try:
        texto = reconhecedor.recognize_google(audio, language='pt-BR')
        print(f"[USUÁRIO] {texto}")
        return texto.lower()
    except sr.UnknownValueError:
        falar("Não entendi, repete aí com calma.")
        return reconhecer_usuario()
    except sr.RequestError:
        falar("Deu ruim no serviço de voz. Tenta de novo depois.")
        return ""
