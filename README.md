# py6sigma

Outils Python open source pour les analyses statistiques Lean Six Sigma.
Cette première version couvre les statistiques descriptives, les tests des
séries et de normalité, la capabilité et quelques graphiques.

## Installation depuis le dépôt

Python 3.10 ou plus récent est nécessaire :

```shell
python -m pip install .
```

## Utilisation

```python
import py6sigma

analysis = py6sigma.Analysis([12, 13, 11, 15, 18, 14, 16, 11, 13, 17])
print(analysis.descriptive_statistics())
print(analysis.run_test())
print(analysis.shapiro_test())
print(analysis.anderson_test())
print(analysis.one_sample_t_test(hypothesized_mean=15))
print(analysis.capability(lsl=10, usl=20))
analysis.capability_analysis_normal(lsl=10, usl=20)
analysis.normality_test()
analysis.run_chart()
print(analysis.run_chart_statistics())
py6sigma.show()
```

### Aide au choix d'une méthode statistique

```python
print(py6sigma.xy_table_choice())
conseil = py6sigma.select_statistical_method(
    y_type="continuous", x_type=None, objective="compare",
    samples=1, distribution="normal",
)
print(conseil.candidates, conseil.missing, conseil.warnings)
```

Le tableau présente des situations et des méthodes candidates. L'aide
au choix distingue notamment les observations appariées des observations
indépendantes et peut demander des informations supplémentaires.
Elle propose des pistes, ne lance aucun test et n'établit
pas que ses hypothèses sont vérifiées. `available` indique si la méthode
proposée est implémentée dans py6sigma, pas si elle est appropriée aux données.
Les termes, paramètres et limites sont détaillés dans
[le guide pour les IA](README_IA.md).

### Test t à un échantillon

```python
import py6sigma

analysis = py6sigma.Analysis([12, 13, 11, 15, 18, 14, 16, 11, 13, 17])
resultat = analysis.one_sample_t_test(
    hypothesized_mean=15,
    alternative="two-sided",
    confidence_level=0.95,
)
print(resultat)
print(resultat.statistic, resultat.degrees_of_freedom, resultat.p_value)
print(resultat.confidence_interval)
```

Le test compare la moyenne de la population à la moyenne hypothétique
fournie. `alternative` vaut `"two-sided"` (différente), `"less"` (inférieure)
ou `"greater"` (supérieure). Le niveau de confiance est une proportion
strictement comprise entre 0 et 1. L'intervalle est bilatéral ou unilatéral
selon l'hypothèse alternative : la borne non limitée vaut `-inf` ou `inf`.
Le résultat fournit aussi N, moyenne, écart-type de l'échantillon et erreur
standard. Le test exige au moins deux valeurs et une dispersion non nulle.
`print(resultat)` présente les résultats sous forme de tableau, avec des
arrondis à l'affichage ; les attributs numériques conservent leur précision.
Une conclusion suit le tableau : elle compare la p-value non arrondie au
seuil `1 - confidence_level` et concerne l'hypothèse alternative choisie.
Elle est valable sous les hypothèses du test.
La colonne importée avec `Analysis.from_excel` nomme automatiquement la
variable ; sinon le nom par défaut est `Data`, modifiable avec `name=`.

Il suppose des observations indépendantes et une population approximativement
normale ; les fortes asymétries et valeurs atypiques sont particulièrement
préoccupantes pour les petits échantillons. Choisir l'hypothèse alternative
avant d'examiner les résultats. Une p-value élevée ne prouve pas l'égalité
des moyennes.
[Méthodes Minitab](https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/how-to/1-sample-t/methods-and-formulas/methods-and-formulas/)
et [options de l'analyse](https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/how-to/1-sample-t/perform-the-analysis/select-the-analysis-options/).

### Graphique de Pareto

```python
import py6sigma

# Effectifs par catégorie :
ax = py6sigma.pareto_chart(
    {"Retard": 42, "Erreur de saisie": 21, "Autre": 7},
    name="Défauts",
)
ax.figure.savefig("pareto.png", dpi=180)
py6sigma.show()
```

Il est aussi possible de fournir des observations brutes :
`py6sigma.pareto_chart(["Rayure", "Bavure", "Rayure"])`, ou deux colonnes
parallèles : `py6sigma.pareto_chart(categories, frequencies=effectifs)`.
Par exemple, pour lire un classeur Excel avec pandas :

```python
import pandas as pd
import py6sigma

donnees = pd.read_excel(r"mes_defauts.xlsx", sheet_name=0)
ax = py6sigma.pareto_chart(donnees["Défaut"], frequencies=donnees["Effectif"])
py6sigma.show()
```

Les catégories doivent être des textes non vides ou des nombres finis ; les
effectifs doivent être des entiers non négatifs et leur somme positive.
Les catégories répétées sont additionnées. Les valeurs manquantes ou
invalides provoquent une erreur : nettoyer les données explicitement avant
l'analyse. La fonction renvoie l'axe Matplotlib et accepte aussi `ax=`.

Les barres sont triées par effectif décroissant ; en cas d'égalité, l'ordre
d'apparition est conservé. Le graphique indique les effectifs, pourcentages
et pourcentages cumulés. Par défaut, après dépassement de 95 % du total,
les catégories restantes sont regroupées dans la dernière barre « Other ».
Utiliser `combine_after=90` (ou `0.9`) pour choisir un autre seuil, et
`combine_after=None` pour afficher toutes les catégories séparément. Si une
catégorie s'appelle déjà « Other », le regroupement utilise un nom distinct.
Les diagrammes par groupe et les Pareto pondérés par coût ne sont pas encore
pris en charge.

Voir les explications Minitab sur [les données et le regroupement](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/pareto-chart/perform-the-analysis/enter-your-data/)
et [l'interprétation du graphique](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/pareto-chart/interpret-the-results/key-results/).

Les méthodes `histogram`, `boxplot`, `run_chart` et `normality_test`
acceptent éventuellement un `matplotlib.axes.Axes` et renvoient cet axe ; elles
n'affichent pas les figures automatiquement.
`py6sigma.show()` affiche les figures avec le backend Matplotlib configuré.

### Scatter plot

```python
import py6sigma

ax = py6sigma.scatter_plot(
    [1, 2, 3, 4], [3, 5, 6, 9],
    x_name="X", y_name="Y",
)
py6sigma.show()
```

Le graphique représente chaque paire `(x, y)` sans droite par défaut.
Les deux séries doivent être numériques, finies, non vides et de même
longueur ; les valeurs manquantes doivent être traitées en conservant
l'alignement des lignes. `groups=...` distingue les catégories, avec
`group_name=...` pour nommer la légende. `regression=True` ajoute une
droite de régression linéaire par groupe (ou une seule sans groupes) ;
chaque groupe doit alors avoir au moins deux valeurs X différentes.
La fonction renvoie un axe Matplotlib et accepte également `ax=`.
Une association sur un scatter plot ne démontre pas une causalité.
Les graphiques pour plusieurs paires de variables et les groupes à plusieurs
niveaux ne sont pas encore pris en charge.

Voir la documentation Minitab sur
[le scatter plot avec groupes](https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/scatterplot/create-the-graph/scatterplot-with-groups/)
et [la droite de régression](https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/scatterplot/create-the-graph/scatterplot-with-regression/).

### Boxplot

`analysis.boxplot(name="Mesure")` affiche la médiane, les quartiles et les
valeurs atypiques. Comme dans le boxplot par défaut de Minitab, les quartiles
sont interpolés aux rangs `(n + 1) × 0,25` et `(n + 1) × 0,75`, limités
aux observations extrêmes. Les moustaches rejoignent les observations
extrêmes situées à au plus 1,5 fois l'écart interquartile de la boîte ;
les valeurs au-delà sont signalées par des astérisques.
Ce calcul des quartiles diffère de celui de `descriptive_statistics()`,
qui utilise l'interpolation par défaut de NumPy. Pour comparer des groupes,
fournir une catégorie pour chaque observation, dans le même ordre :

```python
import py6sigma

analysis = py6sigma.Analysis([2, 3, 4, 5, 6, 8])
analysis.boxplot(name="Mesure", groups=["A", "A", "A", "B", "B", "B"],
                 group_name="Groupe", reference_lines=[4])
py6sigma.show()
```

`reference_lines` ajoute des lignes horizontales aux valeurs indiquées, sur
la même échelle que les mesures ; il est facultatif.
Les groupes sont affichés dans leur ordre d'apparition. Les valeurs de groupe
manquantes ou invalides provoquent une erreur. Les regroupements à plusieurs
niveaux ne sont pas encore pris en charge. Minitab recommande la prudence
pour les petits échantillons, en particulier en dessous de 20 observations.

Voir [les quartiles de Minitab](https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/boxplot/interpret-the-results/quartiles/)
et [l'interprétation du boxplot](https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/boxplot/interpret-the-results/key-results/).

### Importer une colonne Excel

Avec votre propre classeur Excel contenant une colonne numérique `Mesure` :

```python
import py6sigma

analysis = py6sigma.Analysis.from_excel(
    r"mes_donnees.xlsx",
    column="Mesure",
)
analysis.run_chart()
analysis.normality_test()
resultat = analysis.capability(usl=20)
print(resultat)
figure = analysis.capability_analysis_normal(usl=20, name="Mesure")
figure.savefig("capability_analysis_normal.png", dpi=180)
py6sigma.show()
```

`sheet_name` est facultatif : par défaut, la première feuille est utilisée.
Les cellules vides sont ignorées et l'ordre des autres lignes est conservé.
Les valeurs non numériques ou non finies, une colonne absente et un fichier ou
une feuille introuvables provoquent une erreur. Les formules Excel doivent
avoir des valeurs calculées enregistrées dans le classeur.

### Capabilité normale pour mesures individuelles

`capability(lsl=..., usl=...)` renvoie les résultats numériques ;
`capability_analysis_normal(lsl=..., usl=...)` renvoie une figure Matplotlib
avec l'histogramme, les ajustements et les résultats. Fournir au moins une des
deux limites, dans la même unité que les données. Avec une seule limite, Cp et
Pp ne sont pas définis (`None` / `N/A`) ; Cpk et Ppk utilisent la limite
disponible. Les valeurs égales à une limite sont considérées conformes.
Le paramètre facultatif `name` nomme le graphique et son axe horizontal.
Le paramètre `bins` permet de choisir le nombre de classes de l'histogramme,
par exemple `bins=16`. Par défaut, les classes sont choisies automatiquement.
Ce réglage ne modifie pas les indices ni les PPM.

Cette première version traite les **mesures individuelles**, dans leur ordre
de collecte : elle estime l'écart-type court terme par la moyenne des étendues
mobiles de deux observations divisée par 1,128, et l'écart-type global par
l'écart-type de l'échantillon (`ddof=1`). Cp/Cpk utilisent le premier,
Pp/Ppk le second. Les PPM observés sont issus des données ; les PPM attendus
sont calculés sous une hypothèse de loi normale, séparément avec les deux
écarts-types. La stabilité du procédé et l'adéquation du modèle normal doivent
être vérifiées avant d'interpréter ces estimations ; le test de normalité ne
prouve pas à lui seul ces hypothèses. Les sous-groupes de taille supérieure à
un, les transformations, les intervalles de confiance et les indices avec
cible ne sont pas encore pris en charge.

Les méthodes suivent les principes décrits par Minitab pour
[l'estimation des écarts-types](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/methods-and-formulas/methods/)
et la [performance observée](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/methods-and-formulas/observed-performance/).
Les résultats ne sont pas garantis identiques à toutes les options de Minitab.

### Calculer les niveaux sigma à partir des défauts

```python
import py6sigma

resultat = py6sigma.sigma_level(dpo=0.02275)
# Ou : 91 défauts sur 1 000 unités avec 4 opportunités par unité
resultat = py6sigma.sigma_level(defects=91, opportunities_per_unit=4, units=1000)
print(resultat.dpo, resultat.dpmo, resultat.z_lt, resultat.z_st)
```

Le DPO est un taux entre 0 et 1, pas un nombre de défauts par million :
`DPO = défauts / (opportunités par unité × unités)` et `DPMO = DPO × 1 000 000`.
Le calcul suppose le même nombre entier d'opportunités pour chaque unité et
au plus un défaut compté par opportunité. Ne pas mélanger DPO et comptage
de plusieurs défauts pour une même opportunité.

La convention utilisée est `Zlt = Φ⁻¹(1 − DPO)` (queue unilatérale) et
`Zst = Zlt + 1,5`. Elle interprète le DPO fourni comme un taux long terme.
Le décalage de 1,5 est une hypothèse conventionnelle, pas une mesure du
décalage réel ; Zst est donc une estimation conventionnelle. Si le DPO
représente au contraire une performance court terme, la conversion normale
correspond à Zst et il ne faut pas utiliser cette convention telle quelle.
Pour DPO égal à 0 ou 1, les niveaux correspondants sont respectivement
`+inf` ou `-inf` : zéro défaut observé ne prouve pas une performance infinie.
[Référence Minitab sur Z.bench et le niveau sigma](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/supporting-topics/capability-metrics/z-bench-as-an-estimate-of-sigma-capability/).

En ligne de commande, avec une colonne numérique `CycleTime` dans un CSV :

```shell
py6sigma --file data.csv --column CycleTime --lsl 10 --usl 20
```

Les limites sont facultatives et peuvent être fournies séparément.
Utiliser `--no-plots` pour ne pas afficher les graphiques. Les cellules CSV
vides sont ignorées ; les autres valeurs invalides déclenchent une erreur.

## Précisions statistiques

- Les données doivent être finies et à une dimension. L'écart-type descriptif
  est celui de l'échantillon (`ddof=1`) ; l'asymétrie et l'aplatissement de
  données constantes sont indéfinis (`nan`). L'aplatissement est celui de
  Fisher (excès par rapport à la loi normale).
- Le test des séries retire les observations égales à la médiane et utilise
  l'approximation normale de statsmodels, avec sa correction de continuité.
  Il faut au moins trois observations restantes, présentes des deux côtés.
- `run_chart_statistics()` calcule séparément les statistiques présentées sur le
  run chart : séries par rapport à la médiane et séries ascendantes/descendantes,
  nombres attendus et p-values unilatérales approximatives. Les points égaux à
  la médiane sont comptés en dessous ; les valeurs consécutives égales sont
  comptées comme descendantes. La première série ascendante/descendante inclut
  le premier point ; les suivantes commencent après le point de retournement.
  Les p-values utilisent les approximations normales sans correction de
  continuité ; les égalités, notamment consécutives, et les petits échantillons
  peuvent compromettre leur fiabilité. Une statistique impossible à calculer
  est indiquée par `None` dans le résultat et `N/A` sur le graphique.
  Ce calcul diffère donc intentionnellement de `run_test()`.
- Shapiro-Wilk renvoie une statistique et une p-value ; Anderson-Darling renvoie
  sa statistique et les valeurs critiques par niveau de signification, pas de
  p-value. Ne pas rejeter la normalité ne démontre pas qu'elle est vraie.
- Le probability plot représente les observations ordonnées avec les positions
  de rang médian de Benard `(i - 0,3) / (n + 0,4)`, un axe de probabilité
  normale et une droite fondée sur la moyenne
  et l'écart-type de l'échantillon. Dès trois observations, les courbes
  indiquent des intervalles de confiance ponctuels approximatifs à 95 % pour
  les quantiles de la loi normale ajustée (approximation asymptotique, plus
  incertaine pour les petits échantillons). Ce ne sont pas des limites de
  prédiction ni une bande simultanée : des observations en dehors ne prouvent
  pas à elles seules la non-normalité. Dès cinq observations, le graphique
  indique aussi la statistique Anderson-Darling et la p-value approximative de
  `statsmodels.stats.diagnostic.normal_ad`. Les méthodes peuvent différer de
  celles de Minitab ; leurs résultats ne sont pas garantis identiques.
- Les méthodes du run chart et les positions du probability plot sont inspirées
  de la documentation officielle :
  [run chart](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/run-chart/methods-and-formulas/methods-and-formulas/),
  [statistiques du run chart](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/run-chart/interpret-the-results/all-statistics-and-graphs/),
  [positions du probability plot](https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/probability-plot/methods-and-formulas/method-of-obtaining-probability-plot-points/) et
  [test de normalité](https://support.minitab.com/en-us/minitab/help-and-how-to/statistics/basic-statistics/how-to/normality-test/methods-and-formulas/methods-and-formulas/).

## Développement

```shell
python -m pip install -e ".[test]"
python -m pytest
```

Licence : [MIT](LICENSE).
