Package stand-alone - Breakout A3C

Langage / Version / Plateforme
- Langage: Python
- Version recommandee: Python 3.10 a 3.12 (teste avec environnement conda Windows)
- Plateforme: Windows 10/11 (compatible Linux si dependances Atari installees)

Composantes necessaires
- Voir le fichier requirements.txt a la racine.
- Installation (une seule fois) :
  pip install -r requirements.txt

Notes "stand-alone"
- Le package contient le code source, les scripts d'entrainement/test et les checkpoints.
- Pour Atari (Breakout), les composantes ALE/Gymnasium sont necessaires. Selon votre environnement,
  l'acceptation de licence ROM peut demander une etape initiale au premier setup.

Structure principale
- main.py : point d'entree
- train.py / train_dueling.py : entrainement
- test.py / test_dueling.py : evaluation periodique + sauvegarde des resultats
- envs.py : wrappers et creation environnement Atari
- Model/ : checkpoints

Lancer les algorithmes
1) Mode standard (A3C standard)
   python main.py

2) Mode dueling
   python main.py dueling

3) Reprendre depuis un checkpoint specifique
   python main.py standard Model/standard_model_hour_X.pth
   python main.py dueling Model/dueling_model_hour_X.pth

Parametres importants (dans main.py -> class Params)
- lr
- gamma
- tau
- num_processes
- num_steps
- max_episode_length
- stack_frames
- entropy_coef_start / entropy_coef_end / entropy_decay_steps

Sorties
- Logs test: results_standard.txt / results_dueling.txt
- Checkpoint horaire: Model/<type>_model_hour_N.pth
- Meilleur modele (moyenne glissante test): Model/<type>_best.pth

Execution typique (Windows + conda)
1) conda activate <nom_env>
2) python main.py dueling

