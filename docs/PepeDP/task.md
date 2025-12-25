# PepeDP Integration - Task Checklist

## ✅ Implémentation (TERMINÉ)

- [x] Rechercher l'API PepeDP et la fonctionnalité BestTile
- [x] Créer la structure des nodes PepeDP dans `chaiNNer_pytorch/pepedp/`
- [x] Implémenter le node Best Tile (extraction d'un seul tile)
- [x] Implémenter le node Best Tiles (transformer, plusieurs tiles)
- [x] Ajouter PepeDP/Pepeline comme dépendances du package PyTorch
- [x] Créer la catégorie PepeDP dans le package PyTorch
- [x] Ajouter `side_effects=True` au node Best Tile

## ✅ Résolution des Conflits de Versions (TERMINÉ)

- [x] Analyser les conflits de versions (pepedp requiert numpy>=2.2)
- [x] Mettre à jour les dépendances de base de chaiNNer :
  - [x] numpy: 1.24.4 → 2.2.6
  - [x] opencv-python: 4.8.0.76 → 4.12.0.88
  - [x] Pillow: 9.2.0 → 12.0.0
  - [x] scipy: 1.9.3 → 1.16.0
  - [x] numba: 0.57.1 → 0.63.1
  - [x] pymatting: 1.1.10 → 1.1.14
- [x] Supprimer l'environnement Python pour forcer une réinstallation propre

## ⏳ Vérification (EN ATTENTE)

- [ ] Lancer chaiNNer et installer les packages
- [ ] Tester les opérations image de base (Load → Blur → Save)
- [ ] Tester l'upscaling PyTorch
- [ ] Vérifier que Best Tile apparaît dans la catégorie PepeDP
- [ ] Tester Best Tile avec la méthode Laplacian
- [ ] Tester Best Tile avec la méthode IC9600 (GPU)

## 🐛 Bugs à Corriger (PRIORITAIRE)

### Bug 1: Best Tiles n'apparaît pas
- [ ] Vérifier les logs de démarrage pour des erreurs de chargement
- [ ] Vérifier la syntaxe de `best_tiles.py`
- [ ] Vérifier les imports (IteratorInputInfo, IteratorOutputInfo, Transformer)
- [ ] Tester si le node se charge sans erreur

**Fichier** : `/backend/src/packages/chaiNNer_pytorch/pepedp/best_tiles.py`

### Bug 2: Best Tile fonctionne sans output connecté
- [ ] Vérifier que `side_effects=True` est bien la bonne approche
- [ ] Regarder comment Save Image ou d'autres nodes gèrent ça
- [ ] Modifier le node si nécessaire

**Fichier** : `/backend/src/packages/chaiNNer_pytorch/pepedp/best_tile.py`

## 📋 Tests Finaux

- [ ] Tester Best Tiles transformer avec Load Images
- [ ] Vérifier que le filtrage par threshold fonctionne
- [ ] Vérifier que le node est invalide si output non connecté
- [ ] Tester avec des images plus petites que tile_size
- [ ] Tester avec des images exactement de la taille tile_size

## 📁 Fichiers Clés

| Fichier | Description |
|---------|-------------|
| `chaiNNer_standard/__init__.py` | Versions des dépendances de base |
| `chaiNNer_pytorch/__init__.py` | Dépendances PepeDP + catégorie |
| `chaiNNer_pytorch/pepedp/__init__.py` | Groupe Tile |
| `chaiNNer_pytorch/pepedp/best_tile.py` | Node Best Tile |
| `chaiNNer_pytorch/pepedp/best_tiles.py` | Node Best Tiles (transformer) |
