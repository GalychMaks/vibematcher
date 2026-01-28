import torch
import librosa
import numpy as np
from model import E2E
from interference import Inference


def run_extraction(audio_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = E2E(
        hop_length=160,
        n_blocks=4,
        n_gru=1,
        kernel_size=3,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16
    )

    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Помилка завантаження ваг: {e}")

    model.to(device)
    model.eval()

    infer = Inference(
        model=model,
        seg_len=1280,
        seg_frames=160,
        hop_length=160,
        batch_size=16,
        device=device
    )

    audio, sr = librosa.load(audio_path, sr=16000)
    audio_tensor = torch.from_numpy(audio).float().to(device)

    hidden_vecs, pitch = infer.inference(audio_tensor)

    return hidden_vecs.cpu().numpy(), pitch.cpu().numpy()



try:
    print("Обробка першої пісні...")
    vecs1, pitch1 = run_extraction("song1.wav", "rmvpe.pt")

    print("Обробка другої пісні...")
    vecs2, pitch2 = run_extraction("song2.wav", "rmvpe.pt")

    from sklearn.metrics.pairwise import cosine_similarity

    avg_vec1 = vecs1.mean(axis=0).reshape(1, -1)
    avg_vec2 = vecs2.mean(axis=0).reshape(1, -1)

    sim_score = cosine_similarity(avg_vec1, avg_vec2)[0][0]

    print("\n--- РЕЗУЛЬТАТ ПЕРЕВІРКИ ---")
    print(f"Схожість за структурою звуку: {sim_score * 100:.2f}%")

    if sim_score > 0.85:
        print("Висновок: Висока ймовірність плагіату або використання того самого джерела.")
    elif sim_score > 0.6:
        print("Висновок: Є певна схожість, можливо схожий тембр або стиль.")
    else:
        print("Висновок: Пісні оригінальні або дуже різні за звучанням.")

except FileNotFoundError:
    print("Помилка: Переконайся, що файли song1.wav, song2.wav та rmvpe.pt лежать у папці з кодом.")
except Exception as e:
    print(f"Сталася помилка під час виконання: {e}")