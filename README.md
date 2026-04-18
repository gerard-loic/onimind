![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/planche-logo-onimind.jpg?raw=true)

# D'Onitama et de son attrait pour le Deep Learning

Onitama est un jeu de stratégie abstrait, pour deux joueurs, créé par Shimpei Sato et publié en 2014. Il se joue sur un plateau de 5x5 cases, chaque joueur disposant de 5 pions : un maître et 4 disciples. L’objectif est de capturer le maître adverse ou d’amener son propre maître sur le temple de l’adversaire.
La spécificité du jeu réside dans son système de cartes de mouvement. Lors de chaque partie cinq cartes sont tirées aléatoirement parmi un deck de 16 cartes. Ces cartes vont définir les déplacements autorisés pour les pièces du joueur qui les détient, sachant que deux cartes sont attribuées à chaque joueur et qu’une cinquième reste en attente. Dès qu’une carte va être utilisée par un joueur il va échanger la carte utilisée avec celle en attente. Les mouvements légaux évoluent donc en permanence au fil de la partie. 

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/jeu.png?raw=true)

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

L’application de techniques de deep learning au jeu Onitama est peu rependue, cela étant principalement dû au caractère confidentiel du jeu en lui-même, ce qui en fait un sujet d’étude plus original que les traditionnels tic-tac-toe ou puissance 4. Mes recherches ont cependant permis d’identifier sur GitHub deux projets de résolution de la problématique d’Onitama via un réseau neuronal : 
Arijit Dasgupta : https://github.com/arijit-dasgupta/OnitamaAI
Nicolas Maurer : https://github.com/Nicolas-Maurer/Onitama_AlphaZero 

L’approche d’Arijit Dasgupta se base sur un apprentissage en curriculum learning, l’agent montant en niveau en jouant contre des joueurs heuristiques de niveau progressivement plus forts, il n’est pas fait usage de self-play contre le réseau lui-même. L’algorithme d’apprentissage par renforcement utilisé est le D3QN, qui est une extension du DQN, c’est-à-dire une approche basée sur deux réseaux, l’un pour choisir le meilleur coup, l’autre pour l’évaluer. 

Nicolas Maurer choisit pour son projet une implémentation pure d’AlphaZero avec génération de données par self-play. AlphaZero est utilisé pour l’entraînement et l’exploitation du modèle basé sur un réseau de neurones à deux têtes. 

L'approche que je me propose d’adopter dans ce projet se distingue par sa nature progressive et hybride, articulant trois phases successives : un apprentissage supervisé initial à partir de parties jouées, un entraînement par politique proximale (PPO) en self-play, puis une phase d'exploitation où les compétences acquises par le réseau viendront guider des algorithmes de recherche comme Monte Carlo Tree Search et un Minimax sélectif dont l’élagage sera guidé par la politique apprise. 

Là où Dasgupta oppose réseau et heuristiques comme adversaires d'entraînement, et où Maurer fond le MCTS dans la boucle d'apprentissage elle-même, ce projet explore une troisième voie : celle d'un réseau entraîné indépendamment, puis mis au service d'algorithmes de recherche classiques pour en décupler la sélectivité.

# Stratégie d'entraînement

La première étape du projet a consisté à définir puis à matérialiser l'environnement de jeu, c'est à dire un cadre global permettant de jouer une ou des parties avec des joueurs "humains" ou "synthétiques".

Cette première étape réalisée j'ai conçu et développé un ensemble de joueurs "synthétiques" de référence, ceci afin de répondre à deux objectifs : 
- Permettre de générer des données de parties en grand nombre en faisant jouer les joueurs "synthétiques" les uns contre les autres.
- Permettre l'évalution des modèles qui seront réalisés face à des adversaires de différents niveaux.

Différents modèles basés sur des réseaux de neurones, sur base de couches convolutives ou denses et différentes architectures ont été implémentés (Cf. notebook versions-modeles.ipynb). L'ensemble de ces modèles possède une couche d'entrée identique (pour compatibilité avec les données d'entraînement) et deux têtes :
- Policies head : prédiction de politique, ou autrement dit "Dans la situation actuelle, quelles sont les meilleures actions"
- Value head : prédiction de valeur, ou autrement dit "Est ce que la situation actuelle est une bonne ou une mauvaise situation"

L'idée est ensuite d'entraîner les modèles en plusieurs étapes successives : 
- Entraînement supervisé
- Entraînement supervisé de la tête de valeur
- Entraînement PPO

L'évaluation des modèles obtenus permettra de déterminer les capacités des modèles "purs" à atteindre les objectifs fixés et donc leur capacités à résoudre les problématiques de jeu de manière "native", c'est à dire purement via "l'intuition" du réseau.

Ces modèles pourront enfin être mis en combinaison avec d'autres algorithmes de manière à optimiser leurs performances et obtenir des joueurs "mixtes". Ces joueurs seront également évalués afin de déterminer leurs capacités à atteindre les objectifs fixés. 

# Environnement

## Définition
**L'environnement général du jeu est déterminé par :**
- Un **état** actuel
- Le joueur courant (rouge ou bleu)
- La correspondance entre le les joueurs réels et leurs couleurs, ordre de jeu
- Les 16 **cartes** disponibles dans le jeu

**Une carte est déterminée par :**
- Un identifiant numérique unique (de 0 à 15)
- Un nom
- Une couleur (déterminant le premier joueur si c'est carte est la première carte neutre du jeu)
- N **mouvements** relatifs

**Un mouvement d'une carte est déterminé par :**
- Un déplacement relatif selon l'axe x
- Un déplacement relatif selon l'axe y
- Un identifiant unique (de 0 à 51, soit 52 mouvements possibles)

**Un état (C'est à dire l'état du jeu à l'instant T) est déterminé par :**
- Une grille de jeu de 5x5 sur laquelle sont placés : 
    - 4 étudiants du joueur courant
    - 1 maître du jour courant
    - 4 étudiants de l'opposant
    - 1 maître de l'opposant
- 2 cartes disponibles pour le joueur courant
- 2 cartes disponibles pour l'opposant
- 1 carte neutre
Il est important de noter que l'état est toujours représenté du point du vue du joueur courant, le plateau fait donc l'objet d'une translation à chaque coup joué, ceci afin de permettre une meilleure utilisabilité des historiques d'état et exploitation par le réseau.

## Implémentation
Les modules suivants, présents dans le répertoire "onitama/" mettent en oeuvre l'implémentation de l'environnement : 
- game.py   :   Mise en oeuvre d'une partie (classe Game) ou d'un ensemble de N parties (classe GameSession)
- board.py  :   Représentation de l'état actuel (classe Board) et ensemble de fonctions permettant de récupérer les coups possibles, de les jouer et de les annuler
- card.py   :   Implémentation des définitions des cartes et mouvements
- constants :   Constantes générales utilisées dans les différentes classes relatives aux environnements de jeu

## Représentation de l'état pour le réseau

Un état sera représenté sous la forme d'une matrice de 5x5 sur 10 plans (5x5x10) :
- Plan 0 : positions des pions du joueurs courants (0 ou 1)
- Plan 1 : positions des pions de l'adversaire (0 ou 1)
- Plan 2 : position du maître du joueur courant (0 ou 1)
- Plan 3 : position du maître de l'adversaire (0 ou 1)
- Plan 4 : Mouvements de la carte 1 du joueur courant (considérant la position de la pièce en 2:2, 1 pour les destinations relatives possibles)
- Plan 5 : Mouvements de la carte 2 du joueur courant (considérant la position de la pièce en 2:2, 1 pour les destinations relatives possibles)
- Plan 6 : Mouvements de la carte 1 de l'adversaire (considérant la position de la pièce en 2:2, 1 pour les destinations relatives possibles)
- Plan 7 : Mouvements de la carte 2 de l'adversaire (considérant la position de la pièce en 2:2, 1 pour les destinations relatives possibles)
- Plan 8 : Mouvements de la carte neutre (considérant la position de la pièce en 2:2, 1 pour les destinations relatives possibles)
- Plan 9 : Joueur courant premier joueur ou non (Grille de 0 ou de 1)

Cet encodage de l'état a été choisi afin de permettre une représentation "géographique" facilitée des positions, en particulier pour les réseaux à base de couches convolutives. 

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_state_tensor.png?raw=true)

# Joueurs de référence

Plusieurs joueurs non basés sur un réseau de neurones servent de base pour l’évaluation et pour la génération des données de jeu.
- **RandomPlayer** : joueur aléatoire, mesure plancher des performances
- **HeuristicPlayer** : évalue chaque coup possible en appliquant une fonction heuristique.  Le score est basé sur la différence de pièces, la distance du maître au temple adverse, la protection du maître… Différents profils d’heuristiques sont ainsi définis : équilibré, défensif, agressif, favorisant la mobilité, favorisant le contrôle territorial. Le joueur heuristique ne « voit » pas à l’avance et choisit le meilleur coup à un horizon de 1 niveau. 
- **LookAheadHeuristicPlayer** : combine l'heuristique avec un algorithme minimax à profondeur configurable, ce qui lui confère un niveau de jeu significativement plus élevé. (De 1 à 3 niveaux de profondeur, qui correspondront aux joueurs de référence « faible », « moyen » et « fort »)

# Architecture des modèles

## Architecture générale et évolution des versions
Tous les modèles partagent une structure dual-head : un tronc commun qui extrait des représentations de l'état de jeu, puis deux têtes spécialisées :
- **La tête de politique (Policy head)** produit un vecteur de logits de taille 1300 (= 5 colonnes × 5 lignes × 52 déplacements possibles). Lors de la sélection d'un coup, un masque est appliqué pour éliminer les actions illégales avant le softmax. La tête de politique répond à la question « Dans cette situation, quelles sont les meilleures actions à jouer ».
- **La tête de valeur (Value head)** produit un nombre entre -1 et 1 via une activation tanh, estimant la probabilité de victoire depuis la position courante. La tête de valeur répond à la question « Est-ce qu’il s’agit d’une bonne situation ? »
Différentes versions d’architectures de réseau ont été mise en place (récapitulées dans le notebook versions-reseaux-neurones.ipynb)

L’architecture des réseaux de neurones convolutifs s’inspire de celle mise en œuvre dans AlphaGo en utilisant le concept des blocs résiduels. Un bloc résiduel ajoute la sortie d'une ou plusieurs couches à son entrée d'origine (connexion "raccourci"), ce qui permet au réseau d'apprendre uniquement la différence (le résidu) par rapport à ce qu'il avait déjà, rendant l'entraînement de réseaux très profonds beaucoup plus stable.

Voici le récapitulatif versions d'architecture : 

| Vers. | Modèle | Notes | Type | Corps | Tête valeur | Tête politique |
| --- | --- | --- | --- | --- | --- | --- |
| v1 |  | Version initiale. Abandonné -> présente des goulots d'étranglement. | CNN | Conv(128) + 5 blocs residuels (Conv+Conv) | Conv(4)+Dense(64) | Conv(32)+Conv(52) |
| v2 |  | Ajout de métriques additionnelles de validation k-accuracy, utilisation de AdamW au lieu de Adam pour réduire l'over fitting. Abandonné -> architecture avec goulots comme v1 | CNN | Conv(128) + 5 blocs residuels (Conv+Conv) | Conv(4)+Dense(64) | Conv(32)+Conv(52) |
| v3 |  | Architecture réduite par rapport à v2. Abandonné -> architecture avec goulot comme v1 | CNN | Conv(128) + 2 blocs residuels (Conv+Conv) | Conv(4)+Dense(64) | Conv(32)+Conv(52) |
| v4 | Musashi | Version avec inférence tensorflow optimisée et gestion de l'entraînement PPO. Utilisée pour modèle Musashi mais peu optimisée -> présence de goulot | CNN | Conv(128) + 5 blocs residuels (Conv+Conv) | Conv(4)+Dense(64) | Conv(32)+Conv(52) |
| v5 |  | Architecture optimisée pour corriger les problèmes de goulot, utilisé pour modèle Tokugawa | CNN | Conv(128) + 5 blocs residuels (Conv+Conv) | Conv(16)+Dense(128) | Conv(64)+Conv(52) |
| v6 | Kamae | Implémentation de l'entraînement avec masque sur nouvelles données d'entraînement regénérées, utilisé pour modèle Kamae | CNN | Conv(128) + 5 blocs residuels (Conv+Conv) | Conv(16)+Dense(128) | Conv(64)+Conv(52) |
| v7 | Tairanauchu | Premier essai d'architecture sur base de réseau dense, utilisé pour modèle Tairanauchu | Dense | Dense(512)+Dense(512)+Dense(256) | Dense(128)+Dense(64) | Sortie directe 1300 |
| v8 |  | Issu de l'architecture de v6, mais avec 10 couches de blocs résiduels (plus profond) - Abandonné -> trop lourd | CNN | Conv(128) + 10 blocs residuels (Conv+Conv) | Conv(16)+Dense(128) | Conv(64)+Conv(52) |
| v9 | Shigemori | Architecture identique à v7, sur réseau dense, mais avec une couche de plus dans la tête de politique, base du modèle Shigemori | Dense | Dense(512)+Dense(512)+Dense(256) | Dense(128)+Dense(64) | Dense(512) |
| v10 | Sukoshi | Architecture sur réseau dense, sans LayerNormalization, sans Dropout, allégée avec skip connection, base du modèle Sukoshi | Dense | Dense(128)+Dense(128)+Dense(64) | Dense(64)+Dense(32) | Dense(64) |

## Modèle Musashi
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_network_v4_architecture.png?raw=true)

## Modèle Kamae
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_network_v6_architecture.png?raw=true)

## Modèle Tairanauchu
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_network_v7_architecture.png?raw=true)

## Modèle Shigemori
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_network_v9_architecture.png?raw=true)

## Modèle Sukoshi
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/onitama_network_v10_architecture.png?raw=true)

# Génération des données d'entraînement
La première étape du projet consiste à générer un dataset de données en faisant jouer des joueurs de référence de différents niveaux les uns contre les autres. Des données de 30000 parties ont ainsi été générées, seules les trajectoires (ensemble des coups du joueur concerné dans la partie) ayant abouti à une victoire ont été conservées afin de diriger l'apprentissage vers les séquences de jeu les plus efficaces. Pour chaque coup on a conservé l’action jouée, l’état et le masque des coups valides. 

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/performances-joueurs-reference.png?raw=true)

# Entraînement supervisé par imitation
La première phase d'entraînement consiste à apprendre par imitation des joueurs heuristiques de référence. Les différents réseaux ont été entraînés sur la tête de politique uniquement (la tête de valeur est gelée), avec une loss cross-entropie masquée qui contraint le label smoothing aux seuls coups légaux. Sans masque, la softmax distribue de la probabilité sur toutes les 1300 actions, y compris les coups illégaux. Le réseau gaspille alors de la capacité à "raisonner" sur des coups impossibles, et ses probabilités sur les coups légaux sont mécaniquement diluées. 

Durant l’entraînement supervisé j’ai choisi de suivre les métriques de loss, d'accuracy (« le coup joué par le joueur de référence est-il celui prédit ? ») et de top-k accuracy avec k = 3, 5 et 10 (« Le coup joué par le joueur de référence est-il dans les k meilleurs coups prédits ? » ). 

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/entrainement-supervise-kamae.png?raw=true)

Une fois la tête de politique entraînée, une deuxième phase d'entraînement supervisé cible exclusivement la tête de valeur, en gelant le tronc et la tête de politique. Le réseau joue 128 parties par itération contre des joueurs heuristiques, et la tête de valeur est entraînée à prédire le résultat final de la partie (±1) depuis chaque état visité. Cette initialise la tête de valeur à un niveau raisonnable avant le PPO afin de stabiliser l'entraînement par renforcement : sans ce pré-entraînement, les estimations initiales de V(s) sont trop peu fiables pour que le calcul des avantages GAE soit utile dès les premières itérations PPO.

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/performances-apres-supervise.png?raw=true)

# Entraînement PPO

À partir des poids pré-entraînés par imitation, le réseau est affiné par renforcement via l'algorithme PPO. À chaque itération, 128 parties sont collectées en self-play mixte : le réseau affronte à la fois une copie de lui-même (partage de poids) et des joueurs heuristiques de niveaux variés (LookAhead profondeur 1, 2 et 3 avec différentes stratégies) 

Les récompenses sont propagées rétrospectivement via la formule de GAE pour calculer l'avantage estimé de chaque coup. La mise à jour PPO réexploite ensuite ces données en clippant le ratio de politique pour limiter la divergence entre ancienne et nouvelle politique. La loss totale combine la policy loss clippée, une value loss et un bonus d'entropie pour maintenir l'exploration. Autrement dit on réunit trois objectifs distincts en un seul signal de gradient : la policy loss (« Apprendre quoi jouer »), la value loss (« apprendre à s’évaluer ») et l’entropie qui mesure à quel point la distribution des politiques est étalée sur les coups valides. Sans l’entropie le réseau à tendance à converger trop rapidement vers une situation déterministe, c’est-à-dire à jouer toujours les mêmes coups sans laisser sa chance à l’exploration. 


![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/entrainement-PPO.png?raw=true)

Ce cycle est répété pendant 750 itérations, avec une sauvegarde et un test d'efficacité toutes les 10 itérations consistant en une session de parties entre le modèle entraîné dans l’état et les joueurs de référence (Joueur heuristique avec minimax à 1, 2 et 3 niveaux de profondeur). Le taux de victoire (win rate) est mesuré sur 200 parties. 

![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/performances-apres-ppo.png?raw=true)

# Modèles mixtes

Le modèle Kamae obtient après entraînement PPO les meilleures performances puisqu’il est le seul à dépasser la barre symbolique des 50% de win-rate contre un adversaire de niveau moyen. Le niveau atteint est suffisant pour qu’il puisse être utilisé dans le cadre de joueurs « mixtes », c’est-à-dire de joueurs combinant réseau et techniques algorithmiques : 

**Shingen (Kamae + Minimax sélectif)** : les joueurs explorant l’arbre des possibilités avec des règles heuristiques sont efficaces mais montrent leurs limites après une profondeur de 3 niveaux. Les nœuds à visiter sont tellement nombreux que le coût computationnel explose. L’idée pour ce joueur mixte est de combiner le réseau de neurones avec une exploration de l’arbre en minimax mais en favorisant la profondeur de l’arbre plutôt que sa largeur. Ainsi à chaque nœud, seules les 3 meilleures actions évaluées par le modèle sont explorées.

**Nobunaga (Kamae + AlphaZero)** : Si un entraînement du réseau avec l’algorithme AlphaZero représente un coût computationnel inenvisageable dans le cadre de mon projet, la force de cet algorithme est de pouvoir être utilisable également en complément d’un modèle entraîné en phase d’exploitation. L’algorithme AlphaZero permet de parcourir l’arbre des possibilités, guidés en cela par le réseau qui va déterminer par inférence les branches les plus intéressantes à suivre. Le calcul de l’UCB et la découverte progressive de l’arbre va mettre à jour la valeur moyenne des nœuds visités. En définitive l’action la plus intéressante sera celle qui aura été visitée le plus souvent. 

# Evaluation

Une évaluation comparée de l’ensemble des joueurs basés sur des réseaux de neurones contre les joueurs heuristiques de référence permet d’obtenir les résultats d’évaluation suivants : 
![alt text](https://github.com/gerard-loic/onimind/blob/master/notebooks/images/evaluation-finale.png?raw=true)

Ces performances permettent de valider les capacités des modèles entraînés à répondre à la problématique et démontrent la pertinence de la démarche d'entraînement progressive adoptée. En effet, les résultats des modèles "purs" (sans algorithme complémentaire) illustrent l'apport de chaque étape du pipeline : le modèle Kamae, issu de l'architecture convolutive v6, progresse de 85 %/33 %/9 % après l'entraînement supervisé seul jusqu'à 93 %/51 %/27 % après le double entraînement PPO, confirmant que l'apprentissage par renforcement permet au réseau de dépasser significativement le niveau des données d'imitation sur lesquelles il a été initialisé.

Toutefois, les modèles purs restent insuffisants pour valider seuls l'objectif "or", l'adversaire heuristique LookAhead de profondeur 3 représentant une profondeur de calcul que le réseau seul ne peut pas pleinement contrer sans horizon de planification explicite. C'est la combinaison avec des algorithmes de recherche qui permet de franchir ce palier, le réseau servant alors d'oracle d'évaluation et de guidage pour la recherche plutôt que de décideur direct. Les objectifs du projet sont dès lors clairement atteints par les joueurs « mixtes ». 

# Exploitation

Une application de démonstration des modèles entraînés est disponible à l'adresse : http://193.168.144.145:5000
Celle-ci permet à un joueur "humain" de jouer contre les principaux modèles. 

Cette application, réalisée avec Flask est proposée dans le dossier app/ du projet. (python app.py pour l'executer)
Elle nécessite que le service du serveur d'API soit également lancé, celle-ci est dans le dossier api/ du projet. (python app.py pour l'executer)

Documentation de l'API : http://193.168.144.145:8080/docs
