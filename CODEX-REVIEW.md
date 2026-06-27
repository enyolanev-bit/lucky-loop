# CODEX-REVIEW

## 1. Les 3 attaques les plus fortes

1. **Validite statistique du gate historique.** Le critere `effect > max-min inter-seed`
   sur le seul best model n'est pas un test de significativite. Il ignore la variance du
   runner-up, l'appariement par seed/split, l'incertitude due a `n=4`, et le winner's
   curse induit par le fait de choisir le meilleur apres observation. Un reviewer peut
   dire que le Verifier remplace un LLM persuasif par une heuristique arbitraire.

2. **Generalisation experimentale trop faible pour le claim large.** Le resultat cle repose
   sur un petit dataset sklearn, 4 seeds, une metrique accuracy, et 4 methodes classiques.
   Le papier vend une "trust layer" portable, mais l'evidence montre surtout un sanity
   check sur un benchmark jouet. Il faut cadrer la contribution comme mecanisme de
   calibration, pas comme preuve empirique generale sur les agents de recherche.

3. **Langage trop fort autour de "false winners" et "un-foolable".** Les 3 gagnants evites
   sont des claims non supportes, pas des faux positifs etablis au sens statistique fort.
   "Un-foolable" est aussi attaquable : le gate peut etre fool par un protocole biaise,
   des seeds non independantes, du p-hacking sur les configurations, ou une selection de
   benchmark apres coup.

## 2. Fix #1 implemente

J'ai remplace le gate naif dans `luckyworld/src/luckyworld/verifier.py` sans changer
l'API publique `verify(per_method_accs) -> Verdict`.

Nouveau gate :

- classement des methodes par accuracy moyenne, comme avant ;
- comparaison uniquement du best vs 2e ;
- calcul des differences appariees par seed : `acc_best_seed_i - acc_second_seed_i` ;
- IC 95% t de Student sur la moyenne de ces differences ;
- `trustworthy=True` seulement si la borne basse de l'IC95 est strictement > 0.

Le champ existant `seed_noise` reste present pour compatibilite, mais contient maintenant
la marge d'IC95 appariee utilisee par le gate.

## 3. Verification des chiffres du paper

Je n'ai pas pu relancer `noise_sweep.py` de bout en bout dans cet environnement :
`scikit-learn` n'est pas installe. J'ai donc rejoue le nouveau `verify()` sur les accuracies
par seed deja serializees dans `luckyworld/reports/noise_sweep.json`, qui sont les donnees
du tableau du paper.

| Noise | best vs 2e | marge IC95 appariee | borne basse IC95 | Verdict |
|---:|---:|---:|---:|---|
| 0.0 | 0.0035 | 0.0232 | -0.0197 | inconclusive |
| 0.1 | 0.0174 | 0.0264 | -0.0090 | inconclusive |
| 0.2 | 0.0244 | 0.0485 | -0.0240 | inconclusive |
| 0.4 | 0.1066 | 0.0858 | 0.0208 | fiable |

Conclusion : **le headline tient face au gate plus rigoureux** sur les donnees existantes.
Un agent naif annonce 4 gagnants ; le Verifier IC95 confirme 1 gagnant ; donc **3 claims
de gagnant non supportes sont evites**.

Recommandation paper : remplacer "3 false winners avoided" par "3 unsupported winner claims
avoided" si vous voulez etre plus solide face a un reviewer statistique.
