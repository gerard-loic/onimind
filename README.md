# D'Onitama et de son attrait pour le Deep Learning

Onitama est un jeu de stratégie abstrait, pour deux joueurs, créé par Shimpei Sato et publié en 2014. Il se joue sur un plateau de 5x5 cases, chaque joueur disposant de 5 pions : un maître et 4 disciples. L’objectif est de capturer le maître adverse ou d’amener son propre maître sur le temple de l’adversaire.
La spécificité du jeu réside dans son système de cartes de mouvement. Lors de chaque partie cinq cartes sont tirées aléatoirement parmi un deck de 16 cartes. Ces cartes vont définir les déplacements autorisés pour les pièces du joueur qui les détient, sachant que deux cartes sont attribuées à chaque joueur et qu’une cinquième reste en attente. Dès qu’une carte va être utilisée par un joueur il va échanger la carte utilisée avec celle en attente. Les mouvements légaux évoluent donc en permanence au fil de la partie. 

![alt text](https://github.com/gerard-loic/onitama-rl/blob/master/notebooks/images/jeu.png?raw=true)

Ce projet a pour ambition de concevoir un agent artificiel basé sur un réseau de neurones capable d'apprendre à jouer à Onitama. L'objectif final est d'obtenir un agent compétitif, capable de rivaliser avec un joueur humain averti, en ayant développé par lui-même des stratégies cohérentes et efficaces propres à la complexité tactique d'Onitama.

Onitama présente un profil très favorable à l'expérimentation en deep learning pour plusieurs raisons. Il s'agit d'un jeu à information parfaite et entièrement déterministe, où l'état complet du plateau est visible des deux joueurs à tout moment. Le renouvellement des cartes à chaque partie empêche toute mémorisation de séquences et oblige l'agent à généraliser sur des configurations inédites. Son espace d'états, bien que réduit par rapport au Go ou aux échecs grâce à un plateau 5×5, reste suffisamment vaste pour rendre les approches par force brute impraticables en temps réel. En résumé, le jeu est assez complexe pour justifier une approche en deep learning, mais assez contraint pour que l'entraînement reste faisable sur des ressources limitées.

Ce point est un élément important des contraintes appliquées au projet dont les objectifs devront être atteints avec les performances offertes par le matériel à ma disposition, c'est à dire un entraînement sur CPU uniquement. D’autre part les modèles développés devront se baser sur des réseaux de neurones profonds, même si des algorithmes complémentaires pourront être utiliser pour renforcer leur efficacité. (Pas d'utilisation d'heuristique dans ces modèles, l’ « intuition » devra provenir du réseau)

# Objectifs

Voici les objectifs que je me suis fixés dans ce projet d'apprentissage des techniques d'apprentissage profond et par renforcement :

1. Objectif "or"
Développer un joueur basé sur un réseau de neurones profonds, avec ou sans complément d'algorithmes complémentaires, étant de mesure de battre tous les adversaires heuristiques ayant servi à son entraînement. Pour ce faire le réseau doit avoir compris les règles du jeu et développer des stratégies avancées. 

2. Objectif "argent"
Développer un joueur basé sur un réseau de neurones profonds, avec ou sans complément d'algorithmes complémentaires, étant de mesure de battre des adversaires de type heuristiques de niveau moyen. Pour ce faire le réseau doit avoir compris les règles du jeu et développer des stratégies limitées.

3. Objectif "bronze"
Développer un joueur basé sur un réseau de neurones profonds, avec ou sans complément d'algorithmes complémentaires, étant de mesure de battre des adversaires de type heuristiques de niveau faible. Pour ce faire le réseau doit avoir compris les règles du jeu générales et savoir jouer "simplement".

# Etat de l'art

Les jeux de plateau à deux joueurs comme les échecs ou le Go ont toujours occupé une place centrale dans l’histoire de l’intelligence artificielle. Leurs propriétés (informations parfaites, règles déterministes, objectifs clairs) en ont fait des environnements idéaux pour entraîner des agents artificiels dans des conditions contrôlées. 

Les premières approches reposaient sur la force brute et surtout sur des heuristiques (règles d’action) codées à la main dont l’exemple emblématique est la victoire de Deep Blue contre Kasparov en 1997. Le changement de paradigme survient avec TD-Gammon en 1992, premier agent apprenant à jouer par renforcement et s’accélère avec l’essor du deep learning en entraînant la naissance de différents algorithmes appliquées à l’apprentissage par renforcement : 

**AlphaZero**
L’algorithme AlphaZero, applicable à la fois pour l’entraînement et l’exploitation d’un modèle se base sur un réseau de neurones à deux têtes - une tête de politique qui estime la probabilité de chaque coup, et une tête de valeur évaluant la position – qui guide une recherche arborescente de type Monte Carlo Tree Search. Seule la logique de parcours de l’arbre de probabilité est hérité de l’algorithme MCTS, la partie aléatoire est remplacée par l’instinct du réseau. L’algorithme AlphaZero a démontré son efficacité sur de nombreux jeux comme les échecs, cependant l’entraînement d’un réseau en utilisant AlphaZero nécessite des ressources très importantes car chaque coup joué nécessite de nombreuses inférences au réseau.
_"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)_

**MuZero**
L’algorithme MuZero pousse la logique d’AlphaZero encore plus loin : là où AlphaZero connaît les règles du jeu et s’en sert pour construire son arbre de recherche, MuZero apprend lui-même un modèle interne des règles sans qu’on les lui fournisse, directement à partir de l’expérience de jeu. Si MuZero représente l’état de l’art actuel il est au prix d’une complexité d’implémentation et d’un coût computationnel supérieur à AlphaZero. Dans le cas d’Onitama, jeu sur lequel les règles sont connues, son usage n’apporterait pas un bénéfice suffisant. 
_Schrittwieser, J. et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature, 588, 604–609. https://doi.org/10.1038/s41586-020-03051-4_

**Proximal Policy Optimization**
Algorithme d’apprentissage par renforcement de la famille des policy gradient. Le principe est de faire jouer l’agent, d’observer ses résultats, et d’ajuster directement sa politique, c’est-à-dire la manière dont il choisir ses coups afin de favoriser les actions ayant mené à de bons résultats. Le problème de ces méthodes est généralement leur instabilité, une mise à jour trop agressive peut dégrader brutalement les performances et entraîner un cycle de désapprentissage. PPO résout ce problème en introduisant une contrainte limitant l’amplitude de la mise à jour dans un intervalle borné (clipping).
_Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. https://arxiv.org/abs/1707.06347_


# Projets et recherches similaires