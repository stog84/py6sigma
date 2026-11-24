# Guide de sélection statistique pour les outils d'IA

Ce guide décrit l'API de sélection de py6sigma. Elle suggère des méthodes :
elle ne teste pas les données, ne vérifie pas les hypothèses et ne remplace
pas l'analyse du plan d'étude. Ne pas inventer les informations manquantes.

## Vocabulaire

- **Y** : variable réponse étudiée ; son type dépend de ce qu'elle mesure.
- **X** : variable explicative ou variable de groupe ; elle peut être absente
  si un échantillon est comparé à une valeur.
- **Continuous** : mesure quantitative sur une échelle continue, comme une
  durée ; les valeurs peuvent être arrondies dans le fichier.
- **Discrete** : valeur distincte, par exemple catégorie, résultat binaire ou
  décompte. Un décompte numérique ne doit pas automatiquement être traité
  comme une catégorie : ce guide couvre seulement certaines situations.
- **Binary** : variable à deux catégories définies, par exemple conforme ou
  non conforme ; nécessaire pour les tests de proportion de ce guide.
- **Sample/group** : échantillon ou groupe étudié. Compter les groupes selon
  le plan d'étude, pas les catégories observées de Y.
- **Paired** : observations reliées par individu ou unité, par exemple mesures
  avant/après sur les mêmes personnes. Des échantillons de la même population
  ne sont pas nécessairement appariés.
- **Independent** : observations ou groupes sans appariement ni dépendance
  pertinente ; vérifier cette hypothèse avec le plan d'étude.
- **Mean / median** : moyenne / médiane. Un test non paramétrique n'est pas
  nécessairement un test de la moyenne ni même de la médiane.
- **Variance** : mesure de dispersion au carré ; l'égalité des variances
  n'est pas l'égalité des moyennes.
- **Distribution normal/non-normal/unknown** : appréciation du modèle pertinent,
  pas une propriété déduite automatiquement. Pour un t apparié, examiner les
  différences ; pour les comparaisons de groupes, examiner les distributions
  dans chaque groupe. Un test de normalité non significatif ne prouve pas la
  normalité ; la taille d'échantillon et les valeurs atypiques comptent.
- **Estimand** : caractéristique de la population visée par la question
  (moyenne, médiane, proportion, variance, association, etc.).

## Utilisation

```python
import py6sigma

print(py6sigma.xy_table_choice())
result = py6sigma.select_statistical_method(
    y_type="continuous",
    x_type="discrete",
    objective="compare",
    samples=2,
    paired=False,
    distribution="non-normal",
)
for candidate in result.candidates:
    print(candidate.name, candidate.available, candidate.note)
print(result.missing)
print(result.warnings)
```

- `y_type` et `x_type` : `"continuous"` ou `"discrete"` ; `x_type=None`
  signifie qu'aucune variable X n'est fournie.
- `objective` : `"compare"` pour comparer des échantillons continus ;
  `"variance"` pour la dispersion ; `"correlation"` ou `"regression"` pour
  deux variables continues ; `"association"` pour deux variables catégorielles ;
  `"proportion"` pour une proportion binaire.
- `samples` : nombre d'échantillons ou groupes, ou `None` si inconnu.
- `paired` : `True` si les observations sont appariées, `False` si indépendantes,
  `None` si inconnu ou sans objet pour un échantillon.
- `distribution` : `"normal"`, `"non-normal"` ou `"unknown"` ; préciser
  seulement sur la base de l'étude et de l'analyse des données pertinentes.
- `binary` : `True` pour une réponse à deux catégories, `False` sinon, `None`
  si inconnu. Ne pas l'inférer du seul mot « discrete ».
- `candidates` contient des `MethodCandidate` avec `name`, `available` et
  `note`. `available=True` signifie que py6sigma implémente la méthode ;
  ce n'est pas une validation des hypothèses. Le seul test de cette liste
  actuellement implémenté est `Analysis.one_sample_t_test`.
- `missing` contient les informations encore nécessaires. Tant qu'elles ne
  sont pas connues, ne pas annoncer un choix définitif.
- `warnings` signale des restrictions et des situations hors périmètre. Une
  liste vide de candidats ne signifie pas qu'aucune méthode statistique existe.

Le tableau est une synthèse illustrative produite à partir de l'aide au
choix ; il ne remplace pas l'examen des hypothèses et des avertissements.
Mann-Whitney requiert des échantillons indépendants. Pour les données appariées
non normales, l'outil propose des tests sur les différences, avec réserves ;
pour plusieurs mesures appariées ou des variances appariées, il ne tranche pas.
Les tests de rang ne répondent pas automatiquement à une question sur les
moyennes ; interpréter Mann-Whitney ou Kruskal-Wallis comme tests de médianes
requiert notamment des hypothèses sur les distributions.

Les méthodes pour comptes, données ordinales, proportions appariées, modèles
mixtes et plans complexes ne sont pas déterminées automatiquement. Les
petits effectifs, valeurs atypiques, dépendances et multiples comparaisons
peuvent nécessiter d'autres méthodes. Une association n'établit pas une
causalité.

Références : [tests appariés et indépendants](https://support.minitab.com/en-us/workspace/help-and-how-to/forms/types-of-forms/statistical-analysis/2-sample-hypothesis-test/),
[tests de variances](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/anova/supporting-topics/basics/understanding-test-for-equal-variances/)
et [méthode de Mann-Whitney](https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/nonparametrics/how-to/mann-whitney-test/methods-and-formulas/methods-and-formulas/).
