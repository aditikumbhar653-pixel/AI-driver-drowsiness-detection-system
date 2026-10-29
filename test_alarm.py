import os
import pygame
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

alarm_path = os.path.join(
    BASE_DIR,
    "static",
    "alarm.wav"
)

print("Alarm path:")
print(alarm_path)

print("File exists:", os.path.exists(alarm_path))

if not os.path.isfile(alarm_path):
    print("❌ alarm.mp3 सापडली नाही")
    input("Enter दाबा...")
    exit()

try:

    pygame.mixer.init()

    pygame.mixer.music.load(alarm_path)

    print("✅ Alarm Loaded Successfully")

    pygame.mixer.music.play(-1)

    print("🔊 Alarm चालू आहे...")

    time.sleep(10)

    pygame.mixer.music.stop()

    print("🔇 Alarm बंद झाला")


except Exception as error:

    print("❌ Alarm Error:")
    print(error)

input("Enter दाबा...")