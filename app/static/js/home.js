class Home{
    constructor(){
        this.playersContainer = document.getElementById('players');
        this.getPlayers();
    }

    getPlayers(){
        const data = JSON.stringify({});

        const xhr = new XMLHttpRequest();
        xhr.withCredentials = true;

        const self = this;
        xhr.addEventListener('readystatechange', function () {
            if (this.readyState === this.DONE) {
                self.players = JSON.parse(this.responseText);
                self.renderPlayers(players);
            }
        });

        xhr.open('GET', API_URL+'players');
        xhr.setRequestHeader('authorization', 'Basic '+API_KEY);
        xhr.setRequestHeader('content-type', 'application/json');

        xhr.send(data);
    }

    getDifficultyColor(difficulty){
        // 0 = vert clair, 100 = vert foncé
        const lightness = 75 - (difficulty / 100) * 45;
        return `hsl(120, 70%, ${lightness}%)`;
    }

    renderPlayers(players){
        this.playersContainer.innerHTML = '';
        console.log(this.players);
        this.players["players"].forEach(player => {
            const el = document.createElement('div');
            el.className = 'player';

            const name = document.createElement('div');
            name.className = 'name';
            name.textContent = player.name;
            player.difficulty = 100;
            const diff = player.difficulty ?? 0;
            const color = this.getDifficultyColor(diff);
            const angle = (diff / 100) * 360;

            const pie = document.createElement('div');
            pie.className = 'difficulte';
            pie.style.background = `conic-gradient(${color} ${angle}deg, rgba(255,255,255,0.08) ${angle}deg)`;

            const label = document.createElement('div');
            label.className = 'difficulte-label';
            label.textContent = diff;

            pie.appendChild(label);
            el.appendChild(name);
            el.appendChild(pie);

            el.addEventListener('click', () => {
                window.location.href = `/game?player=${encodeURIComponent(player.name)}`;
            });

            this.playersContainer.appendChild(el);
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    home = new Home();
});
