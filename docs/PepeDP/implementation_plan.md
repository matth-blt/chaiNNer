# PepeDP Integration into chaiNNer - Complete Handoff

## 🎯 Objective

Intégrer la fonctionnalité **BestTile de PepeDP** dans chaiNNer pour permettre l'extraction de tiles de haute qualité à partir d'images, utile pour la préparation de datasets ML.

**PepeDP** (https://github.com/umzi2/PepeDP) est une bibliothèque de préparation de datasets créée par umzi2. La fonction BestTile analyse la complexité des images et extrait les régions les plus informatives.

---

## 📝 Résumé de la Conversation

### Phase 1: Recherche et Planification

1. **Analyse de PepeDP** : J'ai étudié le code source de PepeDP pour comprendre son API
   - `LaplacianComplexity` : Méthode CPU rapide basée sur `cv2.Laplacian()`
   - `IC9600Complexity` : Méthode GPU basée sur un réseau de neurones
   - `pepeline.best_tile()` : Fonction Rust qui trouve la position optimale dans une map de complexité

2. **Étude de l'architecture chaiNNer** : J'ai analysé comment créer des nodes dans chaiNNer
   - Les nodes sont des fonctions Python décorées avec `@group.register()`
   - Les packages définissent leurs dépendances dans `__init__.py`
   - Les transformers sont utilisés pour les nodes qui génèrent plusieurs outputs

### Phase 2: Implémentation Initiale

3. **Création d'un package séparé `chaiNNer_pepedp`** (ABANDONNÉ)
   - J'ai d'abord créé un package séparé pour PepeDP
   - Problème : Le package n'était pas découvert par chaiNNer
   - Solution tentée : Ajout manuel dans `server.py` → a fonctionné

4. **Problème de versions des dépendances** (CRITIQUE)
   - PepeDP requiert : `numpy>=2.2`, `numba>=0.61.2`, `llvmlite>=0.44.0`
   - chaiNNer avait : `numpy==1.24.4`, `numba==0.57.1`, etc.
   - L'installation de PepeDP mettait à jour ces packages, puis chaiNNer les réinstallait aux anciennes versions
   - **Conflict de versions en boucle !**

### Phase 3: Résolution des Conflits

5. **Décision : Mettre à jour chaiNNer plutôt que forker PepeDP**
   - L'utilisateur a choisi de mettre à jour les versions par défaut de chaiNNer
   - Analyse d'impact effectuée : le code chaiNNer n'utilise pas d'APIs dépréciées de NumPy 1.x
   - `chainner-ext` (extension Rust) requiert seulement `numpy>=1.16.0` → compatible

6. **Restructuration : PepeDP sous PyTorch**
   - J'ai supprimé le package séparé `chaiNNer_pepedp`
   - J'ai ajouté PepeDP comme dépendances du package PyTorch (évite les doublons)
   - J'ai créé une catégorie "PepeDP" dans le package PyTorch

---

## ✅ Ce Qui a Été Fait

### Fichiers Créés

```
backend/src/packages/chaiNNer_pytorch/pepedp/
├── __init__.py          # Définition du groupe "Tile"
├── best_tile.py         # Node "Best Tile" (extraction d'un seul tile)
└── best_tiles.py        # Node "Best Tiles" (transformer, plusieurs tiles)
```

### Fichiers Modifiés

| Fichier | Modification |
|---------|-------------|
| `chaiNNer_pytorch/__init__.py` | Ajout deps pepedp/pepeline + catégorie PepeDP |
| `chaiNNer_standard/__init__.py` | Mise à jour des versions (voir tableau ci-dessous) |
| `server.py` | Suppression de l'import `chaiNNer_pepedp` (plus nécessaire) |

### Versions Mises à Jour

| Package | Ancienne | Nouvelle |
|---------|----------|----------|
| numpy | 1.24.4 | **2.2.6** |
| opencv-python | 4.8.0.76 | **4.12.0.88** |
| Pillow | 9.2.0 | **12.0.0** |
| scipy | 1.9.3 | **1.16.0** |
| numba | 0.57.1 | **0.63.1** |
| requests | 2.28.2 | **2.32.5** |
| pymatting | 1.1.10 | **1.1.14** |

### Dépendances Ajoutées au Package PyTorch

```python
Dependency(
    display_name="PepeDP",
    pypi_name="pepedp",
    version="0.1.3",
    size_estimate=25 * KB,
),
Dependency(
    display_name="Pepeline",
    pypi_name="pepeline",
    version="1.0.0",
    size_estimate=3 * MB,
),
```

---

## 🐛 Problèmes à Corriger

### Bug 1: Le node "Best Tiles" n'apparaît pas

**Symptôme** : Seul "Best Tile" est visible dans l'interface, pas "Best Tiles".

**Cause probable** :
- Erreur de chargement du node transformer
- Possible problème avec les imports ou les annotations de type

**À vérifier** :
1. Regarder les logs de chaiNNer au démarrage pour des erreurs de chargement
2. Vérifier que le fichier `best_tiles.py` n'a pas d'erreur de syntaxe
3. Vérifier que `IteratorInputInfo` et `IteratorOutputInfo` sont correctement importés

**Fichier concerné** : `/backend/src/packages/chaiNNer_pytorch/pepedp/best_tiles.py`

### Bug 2: Le node "Best Tile" fonctionne sans output connecté

**Symptôme** : Le node s'exécute même si aucun output n'est connecté (ce qui est inutile car les tiles ne sont pas sauvegardés).

**Ce qui a été fait** : J'ai ajouté `side_effects=True` dans la définition du node.

**Problème** : Soit `side_effects=True` ne fonctionne pas comme attendu, soit il y a un autre mécanisme à utiliser.

**À vérifier** :
1. Tester si `side_effects=True` est bien la bonne approche
2. Regarder comment d'autres nodes (comme Save Image) gèrent cette contrainte
3. Peut-être faut-il une approche différente pour les nodes qui produisent des outputs optionnels

**Fichier concerné** : `/backend/src/packages/chaiNNer_pytorch/pepedp/best_tile.py` (ligne 64)

---

## 🔧 État Actuel

### Environnement Python

L'environnement Python de chaiNNer a été **supprimé** pour forcer une réinstallation propre :
```bash
rm -rf "/Users/matthieu/Library/Application Support/chaiNNer/python"
```

### Prochaines Étapes

1. **Lancer chaiNNer** : `npm start` dans `/Users/matthieu/Documents/GitHub/chaiNNer`
2. **Installer les packages** via le Dependency Manager (PyTorch en particulier)
3. **Tester les nodes existants** pour vérifier que l'upgrade n'a rien cassé :
   - Load Image → Gaussian Blur → Save Image
   - Load Image → Upscale Image (PyTorch) → Save Image
4. **Corriger les bugs** :
   - Investiguer pourquoi Best Tiles n'apparaît pas
   - Vérifier/corriger le comportement de side_effects

---

## 📂 Fichiers de Référence

### Code PepeDP Original

```python
# pepedp/scripts/utils/complexity/laplacian.py
class LaplacianComplexity(BaseComplexity):
    def __call__(self, img):
        img = self.image_to_gray(img)
        img = self.median_laplacian(img)
        return np.abs(cv2.Laplacian(img, -1))
```

### Structure du Node Best Tile

```python
@tile_group.register(
    schema_id="chainner:pepedp:best_tile",
    name="Best Tile",
    description=[...],
    icon="BsCrop",
    inputs=[
        ImageInput(),
        NumberInput("Tile Size", ...),
        EnumInput(ComplexityMethod, ...),
        SliderInput("Threshold", ...),
        if_enum_group(2, ComplexityMethod.LAPLACIAN)(
            NumberInput("Median Blur", ...),
        ),
    ],
    outputs=[
        ImageOutput(),
        NumberOutput("Complexity Score", ...),
    ],
    side_effects=True,  # ← Censé exiger un output connecté
)
def best_tile_node(...):
    from pepedp.scripts.utils.complexity.laplacian import LaplacianComplexity
    from pepeline import best_tile
    ...
```

### Structure du Node Best Tiles (Transformer)

```python
@tile_group.register(
    schema_id="chainner:pepedp:best_tiles",
    name="Best Tiles",
    description=[...],
    icon="BsGrid3X3",
    kind="transformer",  # ← C'est un transformer
    inputs=[...],
    outputs=[...],
    iterator_inputs=IteratorInputInfo(inputs=[0], length_type="uint"),
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1], length_type="uint"),
)
def best_tiles_node(...) -> Transformer[np.ndarray, tuple[np.ndarray, float]]:
    ...
    return Transformer(on_iterate=on_iterate)
```

---

## 💡 Notes Importantes

1. **Conversion de couleurs** : PepeDP utilise RGB, chaiNNer utilise BGR. Les nodes font la conversion automatiquement.

2. **Images plus petites que tile_size** : Les nodes retournent l'image entière avec son score de complexité.

3. **Formule dynamic_n_tiles** : `max_tiles = (H * W) // (tile_size² * 2)`

4. **IC9600 nécessite PyTorch/GPU** : L'import est fait de manière lazy dans le node.

---

## 🚀 Pour Continuer

```bash
# 1. Aller dans le dossier chaiNNer
cd /Users/matthieu/Documents/GitHub/chaiNNer

# 2. Lancer l'application
npm start

# 3. Installer PyTorch via Dependency Manager

# 4. Tester et débugger
```

Bonne chance, moi du futur ! 🎉
