import subprocess
import json
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# --- CONFIGURATION ---
PROJECT_ID = "PROJECT ID"      # Votre ID projet
INSTANCE_NAME_BASE = "gpu-worker"
REGION_FILTER = "europe"           # Filtre géographique
MAX_RETRIES = -1                   # -1 = infini
RETRY_DELAY = 120                  # 2 minutes de pause entre les vagues
MAX_WORKERS = 6                    # Nombre de tentatives simultanées

# Mapping GPU -> Machine Type requis
GPU_CONFIG = {
    "nvidia-tesla-t4": "n1-standard-4",
    "nvidia-l4": "g2-standard-4"
}

# Variables globales pour la gestion du parallélisme
stop_event = threading.Event()
print_lock = threading.Lock()

def log(msg):
    """Affiche un message de manière thread-safe."""
    with print_lock:
        print(msg)

def run_gcloud_json(cmd_list):
    """Exécute gcloud et retourne le JSON, gère les erreurs silencieusement pour l'init."""
    try:
        cmd_list = cmd_list + ["--format=json"]
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        return []

def get_zones_for_gpu(gpu_type):
    """Récupère les zones valides et nettoie les URLs pour n'avoir que le nom court."""
    cmd = [
        "gcloud", "compute", "accelerator-types", "list",
        f"--project={PROJECT_ID}",
        f"--filter=name={gpu_type} AND zone:({REGION_FILTER})"
    ]
    data = run_gcloud_json(cmd)

    # CORRECTION MAJEURE : On extrait juste le nom de la zone (ex: 'europe-west1-b')
    # au lieu de l'URL complète ('https://.../europe-west1-b') qui fait planter gcloud create.
    zones = []
    for item in data:
        if 'zone' in item:
            raw_zone = item['zone']
            short_zone = raw_zone.split('/')[-1] # Garde tout ce qui est après le dernier '/'
            zones.append(short_zone)

    return sorted(list(set(zones)))

def create_vm(zone, gpu_type):
    """Tente de créer la VM. S'arrête immédiatement si une autre a réussi."""
    if stop_event.is_set():
        return False

    machine_type = GPU_CONFIG[gpu_type]
    # Nom unique pour éviter les collisions : gpu-worker-t4-europe-west1-b
    instance_name = f"{INSTANCE_NAME_BASE}-{gpu_type.split('-')[-1]}-{zone}"

    log(f"🚀 [Start] Tentative {gpu_type} sur {zone}...")

    cmd = [
        "gcloud", "compute", "instances", "create", instance_name,
        f"--project={PROJECT_ID}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--accelerator=type={gpu_type},count=1",
        "--maintenance-policy=TERMINATE",
        "--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=200GB",
        "--quiet"
    ]

    # Exécution de la commande
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Vérification post-exécution (au cas où quelqu'un a gagné pendant qu'on attendait Google)
    if stop_event.is_set():
        return False

    if result.returncode == 0:
        stop_event.set() # On signale à tout le monde d'arrêter
        log(f"\n✅✅✅ VICTOIRE ! GPU trouvé : {gpu_type} dans {zone}")
        log(f"💻 SSH : gcloud compute ssh {instance_name} --zone={zone}")
        return True
    else:
        # Analyse des erreurs pour le feedback
        err = result.stderr
        if "resources" in err or "exhausted" in err or "not available" in err:
            log(f"🔻 [Fail] {zone} ({gpu_type}): Stock épuisé.")
        elif "quota" in err.lower():
            log(f"❌ [Fail] {zone} ({gpu_type}): Erreur Quota (Vérifiez votre console !).")
        elif "invalid" in err.lower() or "not found" in err.lower():
            # Erreur de syntaxe ou de zone invalide (ne devrait plus arriver avec le fix)
            log(f"⚠️ [Error] {zone}: {err.strip().splitlines()[-1]}")
        else:
            # Erreurs obscures
            pass
        return False

def main():
    attempt_count = 0

    # 1. Découverte des zones (séquentiel)
    log("🔍 Cartographie des zones disponibles (avec nettoyage des noms)...")
    tasks = []
    for gpu_type in GPU_CONFIG.keys():
        zones = get_zones_for_gpu(gpu_type)
        log(f"   -> {gpu_type} détecté dans {len(zones)} zones.")
        for z in zones:
            tasks.append((z, gpu_type))

    if not tasks:
        log("❌ Aucune zone trouvée. Vérifiez le PROJECT_ID ou vos droits.")
        sys.exit(1)

    log(f"🎯 Total combinaisons à tester : {len(tasks)}")

    # 2. Boucle principale de sniping
    while not stop_event.is_set():
        attempt_count += 1
        log(f"\n🏁 --- PASSE N°{attempt_count} (Workers: {MAX_WORKERS}) ---")

        # Mélanger pour répartir la charge et la chance
        random.shuffle(tasks)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for zone, gpu in tasks:
                if stop_event.is_set(): break
                futures.append(executor.submit(create_vm, zone, gpu))

            # On attend que ce batch finisse
            for f in as_completed(futures):
                if stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        if stop_event.is_set():
            break

        if MAX_RETRIES != -1 and attempt_count >= MAX_RETRIES:
            log("\n❌ Echec : Aucun GPU trouvé après le nombre max de tentatives.")
            break

        if not stop_event.is_set():
            log(f"\n💤 Pause de {RETRY_DELAY}s avant nouvelle vague...")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main()
