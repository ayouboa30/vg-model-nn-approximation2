# Rapport de Projet : Option Pricing via Modèle Variance-Gamma et Réseaux de Neurones

Ce document résume le travail effectué pour simuler le "pricing" d'options européennes sous la dynamique Log-Variance-Gamma (VG) en utilisant la méthode de Monte-Carlo (via CUDA C++), et la création d'un modèle de régression par réseau de neurones avec des contraintes de pricing analytiques (via PyTorch).

## 1. Simulation Monte-Carlo (CUDA C++)

### Le Processus Variance-Gamma
Le modèle Variance-Gamma (VG) décrit le prix d'un actif comme une dynamique exponentielle, où le processus sous-jacent est un mouvement brownien évalué en un temps aléatoire dicté par un processus Gamma.
La simulation des processus de Gamma et normaux correspondants nécessite une intégration efficace, qui a été implémentée en CUDA C++ afin de tirer parti de la parallélisation sur GPU.

### Implémentation des Algorithmes
L'implémentation bas-niveau s'appuie sur la bibliothèque `curand` pour la génération de nombres pseudo-aléatoires et se trouve dans `src/cuda_vg/random.cu` et `src/cuda_vg/vg.cu` :
*   **Génération Gamma (Algorithmes 6.7, 6.8)** : Le code gère les différents régimes du paramètre de forme de la distribution Gamma (`a = dt / kappa`). Il utilise l'exponentielle directe pour `a = 1.0`, l'algorithme de Johnk pour `a < 1.0`, et l'algorithme "best" standard pour `a > 1.0`.
*   **Mouvement Brownien (Algorithme 6.11)** : L'implémentation génère les sauts du processus VG $\Delta X$ par le calcul $Z_t = \kappa \cdot \text{Gamma}(\dots)$ suivi de $\theta Z_t + \sigma \sqrt{Z_t} \mathcal{N}(0, 1)$.
*   **Ajustement Martingale & Payoff** : Une fonction (kernel) CUDA distincte pré-calcule d'abord la constante d'ajustement martingale $\omega = \frac{1}{\kappa} \ln(1 - \theta\kappa - \kappa\sigma^2 / 2)$. Le prix de l'option (Call) est ensuite évalué sur des dizaines de milliers de trajectoires via le calcul $\max(0, \exp(\omega T + X_{MC}) - K)$.

*Choix technique* : Pour des performances optimales lors de la simulation par batchs, l'état `curandState` est copié dans la mémoire locale (registres) du thread avant l'exécution de la boucle, évitant de multiples accès coûteux à la mémoire globale du GPU.

## 2. Dataset et Intégration Python (Bindings)

### Nested Monte-Carlo
Plutôt que de générer d'immenses fichiers `.csv` statiques, le pipeline utilise un générateur dynamique : `VGPricingDataset` dans `src/cuda_vg/dataset.py`.
Cette classe hérite de `torch.utils.data.IterableDataset` et connecte la simulation CUDA compilée (`vg.so`) directement à PyTorch en utilisant la librairie standard `ctypes` (`src/cuda_vg/bindings.py`).

*Choix technique* : Ce design (On-The-Fly Data Generation) évite les goulots d'étranglement de l'I/O disque et le manque d'espace de stockage. Lors de l'entraînement, les prix Monte-Carlo, les intervalles de confiance et les variables aléatoires (T, K, $\sigma$, $\theta$, $\kappa$) sont échantillonnés à chaque étape du loader directement dans la VRAM, alimentant le réseau de neurones en flux continu.

## 3. Régression par Réseau de Neurones et Contraintes

L'objectif principal du réseau est d'approximer la fonction d'espérance de Monte-Carlo $\hat{C}(T, K, \kappa, \theta, \sigma) \approx f(T, K, \kappa, \theta, \sigma)$.

Dans le script initial `main.py`, un réseau de type MLP standard était entraîné avec une fonction de perte composite (`CombinedLoss`) essayant de forcer les contraintes de forme (croissance, décroissance, convexité) via des pénalités sur les dérivées du modèle calculées par différentiation automatique (`torch.autograd.grad`).
Bien que ces pénalités fonctionnent (soft-constraints), elles demandent un réglage fastidieux des hyperparamètres et ne garantissent pas mathématiquement les contraintes sur tout le domaine.

### Création du `ConstrainedPricingModel` (PICNN)
Afin de garantir les contraintes de pricing de manière rigoureuse (hard-constraints), nous avons introduit une architecture spécifique inspirée des PICNN (*Partially Input Convex Neural Networks*). L'architecture du modèle (`src/models.py`) intègre ces garanties directement par sa construction :

1.  **Croissance par rapport à la Maturité ($T$)** :
    *   L'entrée $T$ est acheminée à travers des couches `PositiveLinear` (où les poids $W$ sont passés à travers une fonction de softplus : $W^+ = \text{softplus}(W)$).
    *   La fonction d'activation utilisée sur ce chemin est `Tanh`, qui est une fonction strictement croissante.
    *   Par la règle de composition des fonctions croissantes, on s'assure ainsi que le gradient par rapport à $T$ est toujours positif.

2.  **Décroissance par rapport au Strike ($K$)** :
    *   Nous avons créé une nouvelle classe de couche : `NegativeLinear`, où les poids sont calculés comme $W^- = -\text{softplus}(W)$.
    *   Les entrées provenant de $K$ subissent cette transformation négative avant de rejoindre les chemins avec des activations croissantes. Cela garantit un gradient toujours non-positif par rapport à $K$.

3.  **Convexité par rapport au Strike ($K$)** :
    *   Le réseau comporte une voie spécifique $Z$ pour l'état récurrent, dont les couches sont constituées de `PositiveLinear`.
    *   L'activation appliquée à ce chemin de récurrence est la fonction `Softplus` qui est **à la fois convexe et strictement croissante**.
    *   Le paramètre $K$ n'intervient que de manière additive et linéaire (via les poids négatifs). La composée d'une fonction convexe croissante avec une fonction affine préserve la convexité. Par conséquent, la dérivée seconde de l'output par rapport à $K$ est garantie d'être positive ou nulle.

4.  **Positivité du prix** :
    *   L'output final du réseau passe par un ultime `F.softplus`, ce qui garantit qu'un prix d'option sera toujours rendu strictement positif.

### Conclusion sur les Choix d'Architecture
L'architecture contrainte permet un meilleur apprentissage car elle réduit considérablement l'espace de recherche (hypothesis space) aux seules fonctions valides d'un point de vue financier, ce qui accélère l'apprentissage et empêche l'apparition d'opportunités d'arbitrage dans les prédictions du modèle (comme un papillon ("butterfly spread") à prix négatif, ou des anomalies calendaires). La perte est gérée simplement par une pondération spécifique (ex. Huber/MSE pondérée sur l'erreur de Monte-Carlo) sans le besoin coûteux des dérivées dans la fonction objectif.
