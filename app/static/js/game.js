class Onitama{
    constructor(player){
        this.player = player;
        this.board = this.getRef('board');

        this.squares = [];
        this.squares_state = [];
        for(let x=0; x < 5; x++){
            let col = [];
            let col2 = [];
            for(let y=0; y < 5; y++){
                let square = document.createElement('div');
                square.id = x+'_'+y;
                square.className = 'square';
                this.board.appendChild(square);
                let ref = this.getRef(x+'_'+y);
                ref.addEventListener('mouseover', () => this.onMouseOverSquare(x, y));
                ref.addEventListener('mouseout', () => this.onMouseOutSquare(x, y));
                ref.addEventListener('click', () => this.onMouseClickSquare(x, y));
                col.push(ref);
                col2.push(0);
            }
            this.squares.push(col);
            this.squares_state.push(col2);
        }

        let ref = this.getRef("card_red_1");
        ref.addEventListener('mouseover', () => this.onMouseOverCard(1));
        ref.addEventListener('mouseout', () => this.onMouseOutCard(1));
        ref.addEventListener('click', () => this.onMouseClickCard(1));

        ref = this.getRef("card_red_2");
        ref.addEventListener('mouseover', () => this.onMouseOverCard(2));
        ref.addEventListener('mouseout', () => this.onMouseOutCard(2));
        ref.addEventListener('click', () => this.onMouseClickCard(2));

        this.student_red = [
            {'color' : 'red', 'type' : 'student', 'ref' : this.getRef('student_red_1')},
            {'color' : 'red', 'type' : 'student', 'ref' : this.getRef('student_red_2')},
            {'color' : 'red', 'type' : 'student', 'ref' : this.getRef('student_red_3')},
            {'color' : 'red', 'type' : 'student', 'ref' : this.getRef('student_red_4')}
        ];
        this.master_red = {'color' : 'red', 'type' : 'master', 'ref' : this.getRef('master_red')};
        this.student_blue = [
            {'color' : 'blue', 'type' : 'student', 'ref' : this.getRef('student_blue_1')},
            {'color' : 'blue', 'type' : 'student', 'ref' : this.getRef('student_blue_2')},
            {'color' : 'blue', 'type' : 'student', 'ref' : this.getRef('student_blue_3')},
            {'color' : 'blue', 'type' : 'student', 'ref' : this.getRef('student_blue_4')}
        ];
        this.master_blue =  {'color' : 'blue', 'type' : 'master', 'ref' : this.getRef('master_blue')};

        this.log("Onitama game started !");

        this.textes = {
            'red_turn' : "C'est votre tour, choisissez une pièce à déplacer ou une carte.",
            'red_turn_card' : "Choisissez à présent la carte avec laquelle jouer.",
            'red_turn_piece' : "Choisissez à présent la pièce à déplacer.",
            'red_turn_destination' : "Choisissez où déplacer votre pièce.",
            'blue_turn' : "C'est au tour de l'IA...",
            'winner_IA' : "L'IA a gagné la partie !",
            'winner_HUMAN' : "Bravo ! Vous avez gagné la partie !"
        }

        this.createGame();
        
        //Etat actuel de sélection des pièces
        this.playerTurn = false;    //Si c'est le tour du joueur
        this.reinitState();
    }

    reinitState(){
        this.pieceSelected_x = null;  //Pièce sélectionnée (x)
        this.pieceSelected_y = null;  //Pièce sélectionnée (Y)
        this.cardSelected = null;   //Carte sélectionnée
        this.destination_x = null;    //Destination
        this.destination_y = null;    //Destination
        this.square_over_x =  null;
        this.square_over_y = null;
        this.cardOver = null;
        this.destinationSquares = null;
    }

    updateSquares(){
        //Tout déselectionner
        for(let x = 0; x < 5; x++){
            for(let y = 0; y < 5; y++){
                this.squares_state[x][y] = 0;
            }
        }

        //Si pièce sélectionnée
        if(this.pieceSelected_x != null){
            this.squares_state[this.pieceSelected_x][this.pieceSelected_y] = 2;
        }

        //Si case over
        if(this.square_over_x != null){
            if(this.square_over_x != this.pieceSelected_x && this.square_over_y != this.pieceSelected_y){
                this.squares_state[this.square_over_x][this.square_over_y] = 1;
            }
        }

        //Cases de destination
        if(this.destinationSquares != null){
            for(let i=0; i<this.destinationSquares.length; i++){
                if(this.square_over_x == this.destinationSquares[i][0] && this.square_over_y == this.destinationSquares[i][1]){
                    this.squares_state[this.destinationSquares[i][0]][this.destinationSquares[i][1]] = 2;
                }else{
                    this.squares_state[this.destinationSquares[i][0]][this.destinationSquares[i][1]] = 1;
                }
                
            }
        }
        
        //Mise à jour de l'affichage des cases
        for(let x = 0; x < 5; x++){
            for(let y = 0; y < 5; y++){
                let etat = this.squares_state[x][y];
                if(etat == 0){
                    this.squares[x][y].style.opacity = 0;
                    this.squares[x][y].style.backgroundColor = "white";
                }if(etat == 1){
                    this.squares[x][y].style.opacity = 0.5;
                    this.squares[x][y].style.backgroundColor = "white";
                }else if(etat == 2){
                    this.squares[x][y].style.opacity = 0.5;
                    this.squares[x][y].style.backgroundColor = "red";
                }
            }
        }

        //Mise à jour de l'affichage des cartes
        if(this.cardSelected == 1 || this.cardOver == 1){
            this.getRef("card_red_1").style.outline = "5px solid rgba(103, 0, 0, 0.5)";
            this.getRef("card_red_2").style.outline = "5px solid rgba(103, 0, 0, 0)";
        }else if(this.cardSelected == 2 || this.cardOver == 2){
            this.getRef("card_red_1").style.outline = "5px solid rgba(103, 0, 0, 0)";
            this.getRef("card_red_2").style.outline = "5px solid rgba(103, 0, 0, 0.5)";
        }else{
            this.getRef("card_red_1").style.outline = "5px solid rgba(103, 0, 0, 0)";
            this.getRef("card_red_2").style.outline = "5px solid rgba(103, 0, 0, 0)";
        }

    }


    onMouseOverSquare(col, row){
        if(this.playerTurn && !this.current_state.ended){
            //Cas où la pièce n'a pas été sélectionnée, over possible uniquement sur les cases contenant des pièces rouges
            if(this.pieceSelected_x == null){
                let cv = this.board_num[col][row];
                if(cv == this.red_student || cv == this.red_master){
                    this.square_over_x = col;
                    this.square_over_y = row;
                }
            }else if(this.pieceSelected_x != null && this.cardSelected != null){
                for(let i=0; i<this.destinationSquares.length; i++){
                    if(this.destinationSquares[i][0] == col && this.destinationSquares[i][1] == row){
                        this.square_over_x = col;
                        this.square_over_y = row;
                    }
                }
                
            }
            this.updateSquares();
        }
        
    }

    onMouseOutSquare(col, row){
        if(this.playerTurn && !this.current_state.ended){
            if(this.pieceSelected_x != null){
                if(col != this.pieceSelected_x && row != this.pieceSelected_y){
                    this.square_over_x = null;
                    this.square_over_y = null;
                }
            }else{
                this.square_over_x = null;
                this.square_over_y = null;
            }
            this.updateSquares();
        }
        
    }

    onMouseClickSquare(col, row){
        if(this.playerTurn && !this.current_state.ended){
            if(this.pieceSelected_x == null){
                this.pieceSelected_x = col;
                this.pieceSelected_y = row;
                this.checkPlayerChoices();
            }else{
                if(this.pieceSelected_x == col && this.pieceSelected_y == row){
                    this.pieceSelected_x = null;
                    this.pieceSelected_y = null;
                    this.destinationSquares = null;
                    this.checkPlayerChoices();
                }else if(this.pieceSelected_x != null && this.cardSelected != null){
                    for(let i=0; i<this.destinationSquares.length; i++){
                        if(this.destinationSquares[i][0] == col && this.destinationSquares[i][1] == row){
                            this.destination_x = col;
                            this.destination_y = row;
                            
                            let sourcePiece = this.board_num[this.pieceSelected_x][this.pieceSelected_y];
                            this.board_num[col][row] = sourcePiece;
                            this.board_num[this.pieceSelected_x][this.pieceSelected_y] = 0;
                            this.updateBoard(false);
                            this.checkPlayerChoices();
                        }
                    }
                }
            }
            this.updateSquares();
        }
        
    }

    selectDestinationSquares(){
        let card = this.current_state.player_cards[this.cardSelected-1];
        let sx = this.pieceSelected_x;
        let sy = this.pieceSelected_y;

        this.destinationSquares = [];

        for(let i=0; i < card.relative_moves.length; i++){
            let dx = sx + card.relative_moves[i][0];
            let dy = sy + card.relative_moves[i][1];
            if(dx < 0 || dx > 4 || dy < 0 || dy > 4){
                continue;
            }
            if(this.board_num[dx][dy] != this.red_master && this.board_num[dx][dy] != this.red_student){
                this.destinationSquares.push([dx, dy]);
            }
        }
    }

    

    onMouseOverCard(i){
        if(this.playerTurn){
            if(this.cardSelected == null){
                this.cardOver = i;
            }
            this.updateSquares();
        }
        
    }

    onMouseOutCard(i){
        if(this.playerTurn){
            if(this.cardSelected == null){
                this.cardOver = null;
            }
            this.updateSquares();
        }
        
    }

    onMouseClickCard(i){
        if(this.playerTurn){
            if(this.cardSelected == null){
                this.cardSelected = i;
            }else{
                this.cardSelected = null;
                this.destinationSquares = null;
            }
            this.checkPlayerChoices();
            this.updateSquares();
        }
        
        
    }

    checkPlayerChoices(){
        if(this.playerTurn && !this.current_state.ended){
            if(this.pieceSelected_x == null && this.cardSelected == null && this.destination_x == null){
                this.talk(this.textes['red_turn']);
            }else if(this.pieceSelected_x != null && this.cardSelected == null && this.destination_x == null){
                this.talk(this.textes['red_turn_card']);
            }else if(this.pieceSelected_x == null && this.cardSelected != null && this.destination_x == null){
                this.talk(this.textes['red_turn_piece']);
            }else if(this.pieceSelected_x != null && this.cardSelected != null && this.destination_x == null){
                this.talk(this.textes['red_turn_destination']);
                this.selectDestinationSquares();
            }else{
                this.talk("OK ! Patientez, c'est au tour de l'IA...");
                this.play();
            }
        }
    }

    setWinner(){
        if(this.current_state.winner == "IA"){
            this.talk(this.textes['winner_IA']);
        }else{
            this.talk(this.textes['winner_HUMAN']);
        }
    }

    createGame(){
        this.log("Creating the game...");

        const params = new URLSearchParams(window.location.search);
        const playerUid = params.get('player') || 'heuristic_3lookahead_regular';

        const data = JSON.stringify({
        player: playerUid
        });

        const xhr = new XMLHttpRequest();
        xhr.withCredentials = true;

        let ref = this;
        xhr.addEventListener('readystatechange', function () {
        if (this.readyState === this.DONE) {
            console.log(this.responseText);
            ref.current_state = JSON.parse(this.responseText);
            ref.uid = ref.current_state.game_uid
            ref.humanIsPlayerOne = (ref.current_state.current_player == "HUMAN");
            if(ref.current_state.current_player == "IA"){
                ref.updateBoard(true, true);
                ref.opponentPlay();
            }else{
                ref.updateBoard(true);
            }
            
        }
        });

        xhr.open('POST', API_URL+'game');
        xhr.setRequestHeader('authorization', 'Basic '+API_KEY);
        xhr.setRequestHeader('content-type', 'application/json');

        xhr.send(data);
    }

    play(){
        const data = JSON.stringify({
        from_pos_col: this.pieceSelected_x,
        from_pos_row: this.pieceSelected_y,
        to_pos_col: this.destination_x,
        to_pos_row: this.destination_y,
        card_idx: this.current_state.player_cards[this.cardSelected-1]['idx']
        });

        const xhr = new XMLHttpRequest();
        xhr.withCredentials = true;

        let ref = this;
        xhr.addEventListener('readystatechange', function () {
        if (this.readyState === this.DONE) {
            console.log(this.responseText);
            ref.current_state = JSON.parse(this.responseText);


            if(ref.current_state.ended){
                console.log("ENDED !!!!!!");
                ref.setWinner();
                ref.reinitState();
                ref.updateSquares();
                //ref.updateBoard(true, true);
            }else{
                ref.updateBoard(true, true);
                ref.reinitState();
                ref.updateSquares();
                ref.opponentPlay();
            }
            
        }
        });

        xhr.open('POST', API_URL+'game/'+ref.uid+'/player/play');
        xhr.setRequestHeader('authorization', 'Basic '+API_KEY);
        xhr.setRequestHeader('content-type', 'application/json');

        xhr.send(data);
    }

    opponentPlay(){
        console.log("IA PLAY");
        const data = JSON.stringify({});

        const xhr = new XMLHttpRequest();
        xhr.withCredentials = true;

        let ref = this;
        xhr.addEventListener('readystatechange', function () {
        if (this.readyState === this.DONE) {
            console.log("REPONSE");
            console.log(this.responseText);
            

            let temp_state = JSON.parse(this.responseText);
            for(let i = 0; i < ref.current_state.player_cards.length; i++){
                if(ref.current_state.player_cards[i].idx == temp_state.last_move.card_idx){
                    ref.getRef("card_blue_"+(i+1)).style.outline = "5px solid rgba(103, 0, 0, 0.5)";
                }
            }

            setTimeout(() => {
                //ref.reinitState();
                //ref.updateBoard(true);
                //ref.updateSquares();

                //From pos
                let rotate_from_pos_col = 4 - temp_state.last_move.from_pos.col
                let rotate_from_pos_row = 4 - temp_state.last_move.from_pos.row
                ref.squares[rotate_from_pos_col][rotate_from_pos_row].style.opacity = 0.5;
                ref.squares[rotate_from_pos_col][rotate_from_pos_row].style.backgroundColor = "red";

                setTimeout(() => {
                    //ref.reinitState();
                    //ref.updateBoard(true);
                    //ref.updateSquares();

                    let rotate_to_pos_col = 4 - temp_state.last_move.to_pos.col
                    let rotate_to_pos_row = 4 - temp_state.last_move.to_pos.row
                    ref.squares[rotate_to_pos_col][rotate_to_pos_row].style.opacity = 0.5;
                    ref.squares[rotate_to_pos_col][rotate_to_pos_row].style.backgroundColor = "red";

                    ref.current_state = JSON.parse(this.responseText);
                    ref.updateBoard(false, true, false);

                    ref.getRef("card_blue_1").style.outline = "5px solid rgba(103, 0, 0, 0)";
                    ref.getRef("card_blue_2").style.outline = "5px solid rgba(103, 0, 0, 0)";
                    ref.reinitState();
                    ref.updateBoard(true);
                    ref.updateSquares();

                    if(ref.current_state.ended){
                        console.log("IA WIN !");
                        ref.setWinner();
                        //ref.updateBoard(true);
                        //ref.updateSquares();
                    }

                }, 1000);

            }, 1000);

            //ref.current_state = JSON.parse(this.responseText);

            //this.getRef("card_blue_1").style.outline = "5px solid rgba(103, 0, 0, 0)";
            //this.getRef("card_blue_2").style.outline = "5px solid rgba(103, 0, 0, 0)";


            //setTimeout(() => {
                //ref.reinitState();
                //ref.updateBoard(true);
                //ref.updateSquares();
            //}, 1000);
        }
        });

        xhr.open('POST', API_URL+'game/'+ref.uid+'/opponent/play');
        xhr.setRequestHeader('authorization', 'Basic '+API_KEY);
        xhr.setRequestHeader('content-type', 'application/json');

        xhr.send(data);
    }



    updateBoard(newTurn, opponent = false, updateCards = true){
        this.calculPositions();

        // Copie profonde pour ne pas muter current_state.board
        this.board_num = this.current_state.board.map(col => [...col]);

        this.red_student = 1;
        this.red_master = 2;
        this.blue_student = 3;
        this.blue_master = 4;

        //Dabord masquer toutes les pièces.
        this.student_red[0]["ref"].style.display = "none";
        this.student_red[1]["ref"].style.display = "none";
        this.student_red[2]["ref"].style.display = "none";
        this.student_red[3]["ref"].style.display = "none";
        this.student_blue[0]["ref"].style.display = "none";
        this.student_blue[1]["ref"].style.display = "none";
        this.student_blue[2]["ref"].style.display = "none";
        this.student_blue[3]["ref"].style.display = "none";
        this.master_red["ref"].style.display = "none";
        this.master_blue["ref"].style.display = "none";

        if(!this.humanIsPlayerOne){
            this.red_student = 3;
            this.red_master = 4;
            this.blue_student = 1;
            this.blue_master = 2;
        }

        // Le backend retourne toujours le plateau du point de vue du joueur courant.
        // Quand c'est le tour de l'IA, le plateau est dans sa perspective (pièces IA en bas).
        // On le retourne pour toujours afficher du point de vue de l'humain (pièces humain en bas).
        if(this.current_state.current_player == "IA"){
            this.board_num.reverse().forEach(row => row.reverse());
        }

        let red_i = 0;
        let blue_i = 0;
        for(let x=0; x < 5; x++){
            for(let y=0; y < 5; y++){
                let board_cel = this.board_num[x][y]
                if(board_cel == this.red_student){
                    console.log("STUDENT");
                    this.placePiece(this.student_red[red_i]["ref"], "student", x, y);
                    red_i++;
                }else if(board_cel == this.blue_student){
                    this.placePiece(this.student_blue[blue_i]["ref"], "student", x, y);
                    blue_i++;
                }else if(board_cel == this.red_master){
                    this.placePiece(this.master_red["ref"], "master", x, y);
                }else if(board_cel == this.blue_master){
                    this.placePiece(this.master_blue["ref"], "master", x, y);
                }
            }
        }

        for(let i = red_i; i < 4; i++){
            this.student_red[i]["ref"].style.display = "None";
        }
        for(let i = blue_i; i < 4; i++){
            this.student_blue[i]["ref"].style.display = "None";
        }

        //On met en place les cartes
        if(updateCards){
            if(this.current_state.current_player == "IA"){
                this.getRef("card_blue_1").style.backgroundImage = "url('/static/images/carte_"+this.current_state.player_cards[0].idx+".png')";
                this.getRef("card_blue_2").style.backgroundImage = "url('/static/images/carte_"+this.current_state.player_cards[1].idx+".png')";
                this.getRef("card_red_1").style.backgroundImage = "url('/static/images/carte_"+this.current_state.opponent_cards[0].idx+".png')";
                this.getRef("card_red_2").style.backgroundImage = "url('/static/images/carte_"+this.current_state.opponent_cards[1].idx+".png')";
                this.getRef("card_neutral").style.backgroundImage = "url('/static/images/carte_"+this.current_state.neutral_card.idx+".png')";
            }else{
                this.getRef("card_red_1").style.backgroundImage = "url('/static/images/carte_"+this.current_state.player_cards[0].idx+".png')";
                this.getRef("card_red_2").style.backgroundImage = "url('/static/images/carte_"+this.current_state.player_cards[1].idx+".png')";
                this.getRef("card_blue_1").style.backgroundImage = "url('/static/images/carte_"+this.current_state.opponent_cards[0].idx+".png')";
                this.getRef("card_blue_2").style.backgroundImage = "url('/static/images/carte_"+this.current_state.opponent_cards[1].idx+".png')";
                this.getRef("card_neutral").style.backgroundImage = "url('/static/images/carte_"+this.current_state.neutral_card.idx+".png')";
            }
        }
        
        
        if(this.current_state.current_player == "IA"){
            this.talk(this.textes['blue_turn']);
        }else{
            this.talk(this.textes['red_turn']);
        }

        if(newTurn){

            if(this.current_state.current_player == "IA"){
                this.playerTurn = false;
            }else{
                this.playerTurn = true;
            }
            this.pieceSelected_x = null;  //Pièce sélectionnée (x)
            this.pieceSelected_y = null;  //Pièce sélectionnée (Y)
            this.cardSelected = null;   //Carte sélectionnée
            this.destination_x = null;    //Destination
            this.destination_y = null;    //Destination
        }
    }

    calculPositions(){
        //Calcul des coordonnées
        this.board_width = this.board.clientWidth;
        this.board_height = this.board.clientHeight;
        this.student_width = this.student_red[0]["ref"].clientWidth;
        this.student_height = this.student_red[0]["ref"].clientHeight;

        this.master_width = this.master_red["ref"].clientWidth || this.master_blue["ref"].clientWidth;
        this.master_height = this.master_red["ref"].clientHeight || this.master_blue["ref"].clientHeight;

        this.width_margin = this.board_width*0.03;
        this.height_margin = this.board_height*0.22;

        this.a_x = this.width_margin;
        this.a_y = this.height_margin;
        this.b_x = this.board_width-this.width_margin;
        this.b_y = this.height_margin;
        this.c_x = this.width_margin;
        this.c_y = this.board_height-this.height_margin;
        this.d_x = this.board_width-this.width_margin;
        this.d_y = this.board_height-this.height_margin;

        this.positions = [];
        this.case_width = (this.b_x - this.a_x) / 5;
        this.case_height = (this.c_y - this.a_y) / 5;
        for(let x=0; x < 5; x++){
            let col = []
            for(let y=0; y < 5; y++){
                col.push({'x' : (this.width_margin + (this.case_width*x)), 'y' : (this.height_margin + (this.case_height*y))});

                this.squares[x][y].style.top = (this.height_margin + (this.case_height*y))+ 'px';
                this.squares[x][y].style.left = (this.width_margin + (this.case_width*x))+ 'px';
                this.squares[x][y].style.zIndex = 1;
            }
            this.positions.push(col);
        }
    }

    placePiece(piece, type, col, row){
        piece.style.display = "";
        let piece_w = this.student_width;
        let piece_h = this.student_height;

        if(type == "master"){
            piece_w = this.master_width;
            piece_h = this.master_height;
        }

        let x = this.positions[col][row]['x'] + ((this.case_width - piece_w) / 2);
        let y = this.positions[col][row]['y'] - (this.case_height / 4);

        piece.style.top = y + 'px';
        piece.style.left = x + 'px';
        piece.style.zIndex = 10 + col + row;
        
    }

    talk(texte){
        this.getRef("parole").innerHTML = texte;
    }

    getRef(name){
        return document.getElementById(name);
    }

    log(element){
        console.log(element);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    game = new Onitama("test");
});
